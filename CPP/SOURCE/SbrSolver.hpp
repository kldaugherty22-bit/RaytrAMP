#ifndef SBR_SOLVER_INCLUDED
#define SBR_SOLVER_INCLUDED

#include <complex>

#include "TypeDef.hpp"
#include "Triangle.hpp"
#include "TriangleMesh.hpp"
#include "MortonManager.hpp"
#include "BvhNodeTypes.hpp"
#include "ReducedBvhArray.hpp"
#include "RayPool.hpp"
#include "RcsArray.hpp"
#include "ObservationArray.hpp"

template< class T >
class SbrSolver
{

public:
	static T c0;
	static T mu0;
	static T eps0;
	static T z0;
	static T pi;
	
public:

	SbrSolver()
	{

	}

	~SbrSolver()
	{

	}

	void MonostaticRcsGpu( const ReducedBvhArray< T >& bvhArray, const ObservationArray< T >& obsArray, RcsArray< T >& rcsArray )
	{
		U32 obsCount = obsArray.obsCount_;
		Observation< T >* obsPtr = obsArray.observationArray_.get();
		T* rcsPtr = rcsArray.rcsArray_.get();
		RayPool< T > rayPool;

		for( U32 idx = 0; idx < obsCount; ++idx )
		{
			PopulateRayPool( bvhArray, obsPtr[ idx ], rayPool );
			ShootAndBounceRaysGpu( bvhArray, rayPool );
			PhysicalOpticsIntegral( rayPool, obsPtr[ idx ], rcsPtr[ idx ] );
		}
	}

	void PopulateRayPool( const ReducedBvhArray< T >& bvhArray, const Observation< T >& observation, RayPool< T >& rayPool )
	{
		BoundBox< T > boundBox = bvhArray.bvhNodeArray_.get()[ 0 ].data_.boundBox_;
		T lambdaWidthMeter = c0 / observation.frequency_;
		T objWidthMeter = boundBox.GetRadius() * 2;
		T objWidthLambda = objWidthMeter / lambdaWidthMeter;
		U32 rayCountSqrt = (U32)( std::ceil( objWidthLambda * (T)( observation.rayPerLam_ ) + 1.0 ) ) + 1;
		rayPool.Initialize( rayCountSqrt );
		rayPool.ReGenerateRays( boundBox, observation.direction_, observation.polarization_ );
	}

	void ShootAndBounceRaysGpu( const ReducedBvhArray< T >& bvhArray, RayPool< T >& rayPool )
	{
		using namespace concurrency;

		ReducedBvhNode< T >* bvhPtr = bvhArray.bvhNodeArray_.get();
		RayTube< T >* rayPtr = rayPool.rayTubeArray_.get();
		
		array_view< const ReducedBvhNode< T >, 1 > bvhGpu( bvhArray.nodeCount_, bvhPtr );
		array_view< RayTube< T >, 1 > rayGpu( rayPool.rayCount_, rayPtr );

		parallel_for_each( rayGpu.extent, [=]( index< 1 > idRay ) restrict( amp )
		{
			RayTube< T >& ray = rayGpu[ idRay ];
			bool isHitAtAll = true;

			ray.dist_ = 0;

			const T epsBasis = (T)1e-8;

			while( isHitAtAll && ray.refCount_ < 10 )
			{
				isHitAtAll = false;

				// Increased from 32 to 64 and protected below.
				U32 stackArray[ 64 ];
				I32 stackIdx = 0;
				stackArray[ stackIdx ] = 0;

				T hitDistMin = (T)1E32;
				U32 hitIdx = (U32)-1;
				LUV::Vec3< T > hitPointMin;

				while( stackIdx >= 0 )
				{
					bool isHit = false;
					T hitDist = 0;

					const ReducedBvhNode< T >& node = bvhGpu[ index< 1 >( stackArray[ stackIdx ] ) ];

					if( node.status_ == BRANCH || node.status_ == ROOT )
					{
						ray.CollisionWithBoundBox( node.data_.boundBox_, isHit, hitDist );
						if( isHit && hitDist < hitDistMin )
						{
							// Prevent stack overwrite.
							if( stackIdx + 2 >= 64 )
							{
								isHitAtAll = false;
								break;
							}

							stackArray[ stackIdx++ ] = node.data_.leftChildIdx_;
							stackArray[ stackIdx++ ] = node.data_.rightChildIdx_;
						}
					}
					else if( stackArray[ stackIdx ] != ray.lastHitIdx_ )
					{
						LUV::Vec3< T > hitPoint;
						ray.CollisionWithTriangleSbr( node.trig_, isHit, hitDist, hitPoint );
						if( isHit && hitDist < hitDistMin )
						{
							isHitAtAll = true;
							hitDistMin = hitDist;
							hitPointMin = hitPoint;
							hitIdx = stackArray[ stackIdx ];
						}
					}

					--stackIdx;
				}

				if( isHitAtAll )
				{
					const ReducedBvhNode< T >& node = bvhGpu[ index< 1 >( hitIdx ) ];
					LUV::Vec3< T > hitNormal = node.trig_.GetNormal();

					// Build a stable polarization basis.
					LUV::Vec3< T > dirCrossNormal = LUV::Cross( ray.dir_, hitNormal );
					T dirCrossNormalLen = LUV::Length( dirCrossNormal );

					LUV::Vec3< T > polU;
					if( dirCrossNormalLen > epsBasis )
					{
						polU = dirCrossNormal / dirCrossNormalLen;
					}
					else
					{
						// Ray direction is parallel/nearly parallel to the surface normal.
						// Pick a fallback axis that is not parallel to ray.dir_.
						LUV::Vec3< T > helper;
						if( fabs( ray.dir_[2] ) < (T)0.9 )
						{
							helper = LUV::Vec3< T >( (T)0.0, (T)0.0, (T)1.0 );
						}
						else
						{
							helper = LUV::Vec3< T >( (T)0.0, (T)1.0, (T)0.0 );
						}

						LUV::Vec3< T > fallbackCross = LUV::Cross( helper, ray.dir_ );
						T fallbackLen = LUV::Length( fallbackCross );

						// If this somehow still degenerates, abandon this bounce safely.
						if( fallbackLen <= epsBasis )
						{
							isHitAtAll = false;
							break;
						}

						polU = fallbackCross / fallbackLen;
					}

					LUV::Vec3< T > polRCross = LUV::Cross( ray.dir_, polU );
					T polRLen = LUV::Length( polRCross );
					if( polRLen <= epsBasis )
					{
						isHitAtAll = false;
						break;
					}
					LUV::Vec3< T > polR = polRCross / polRLen;

					LUV::Vec3< T > refDirRaw = ray.dir_ - hitNormal * ( (T)2.0 * LUV::Dot( ray.dir_, hitNormal ) );
					T refDirLen = LUV::Length( refDirRaw );
					if( refDirLen <= epsBasis )
					{
						isHitAtAll = false;
						break;
					}
					LUV::Vec3< T > refDir = refDirRaw / refDirLen;

					LUV::Vec3< T > refPolU = -polU;

					LUV::Vec3< T > refPolRCross = LUV::Cross( refDir, refPolU );
					T refPolRLen = LUV::Length( refPolRCross );
					if( refPolRLen <= epsBasis )
					{
						isHitAtAll = false;
						break;
					}
					LUV::Vec3< T > refPolR = refPolRCross / refPolRLen;

					T polCompU = LUV::Dot( ray.pol_, polU );
					T polCompR = LUV::Dot( ray.pol_, polR );

					ray.pos_ = hitPointMin;
					ray.dir_ = refDir;
					ray.pol_ = - polCompR * refPolR + polCompU * refPolU;
					ray.dist_ += hitDistMin;
					ray.refNormal_ = hitNormal;
					ray.refCount_ += 1;
					ray.lastHitIdx_ = hitIdx;
				}
			}
		});
		rayGpu.synchronize();
	}

	void PhysicalOpticsIntegral( const RayPool< T >& rayPool, const Observation< T >& obs, T& rcs )
	{
		using namespace std;
		using namespace std::complex_literals;

		T freq = obs.frequency_;
		T angFreq = 2 * pi * freq;
		T waveLen = c0 / freq;
		T waveNum = 2 * pi / waveLen;

		RayTube< T >* rayPtr = rayPool.rayTubeArray_.get();
		U32 rayCount = rayPool.rayCount_;
		T rayArea = rayPool.rayArea_;

		LUV::Vec3< T > obsDir = obs.direction_;
		LUV::Vec3< T > obsDirSph = LUV::CtsToSph( obsDir );

		T phi = obsDirSph[ 1 ];
		T the = obsDirSph[ 2 ];

		T cp = cos( phi );
		T sp = sin( phi );
		T ct = cos( the );
		T st = sin( the );

		LUV::Vec3< T > dirX( 1.0, 0.0, 0.0 );
		LUV::Vec3< T > dirY( 0.0, 1.0, 0.0 );
		LUV::Vec3< T > dirZ( 0.0, 0.0, 1.0 );
		LUV::Vec3< T > dirP( -sp, cp, 0.0 );
		LUV::Vec3< T > dirT( cp * ct, sp * ct, -st );

		LUV::Vec3< T > vecK = waveNum * ( ( dirX * cp + dirY * sp ) * st + dirZ * ct );
		
		complex< T > AU = 0;
		complex< T > AR = 0;

		complex< T > i( 0.0, 1.0 );

		for( U32 idRay = 0; idRay < rayCount; ++idRay )
		{
			RayTube< T >& ray = rayPtr[ idRay ];
			if( ray.refCount_ > 0 )
			{
				T kr = waveNum * ray.dist_;
				//T reflectionCoef = pow( -1.0, ray.refCount_ );
				T reflectionCoef = pow( 1.0, ray.refCount_ );

				LUV::Vec3< complex< T > > apE = exp( i * kr ) * ray.pol_ * reflectionCoef;
				LUV::Vec3< complex< T > > apH = -LUV::Cross( apE, ray.dir_ );

				complex< T > BU = LUV::Dot( -( LUV::Cross( apE, -dirP ) + LUV::Cross( apH, dirT ) ), ray.dir_ );
				complex< T > BR = LUV::Dot( -( LUV::Cross( apE, dirT ) + LUV::Cross( apH, dirP ) ), ray.dir_ );

				complex< T > factor = complex< T >( 0.0, ( ( waveNum * rayArea ) / ( 4.0 * pi ) ) ) * exp( -i * LUV::Dot( vecK, ray.pos_ ) );

				AU += BU * factor;
				AR += BR * factor;
			}

		}

		//std::cout << "AU: " << AU.real() << " + i" << AU.imag() << std::endl;
		//std::cout << "AR: " << AR.real() << " + i" << AR.imag() << std::endl;

		rcs = 4.0 * pi * ( pow( abs( AU ), 2 ) + pow( abs( AR ), 2 ) ); // * 4 * pi

	}



};


template< class T >
T SbrSolver< T >::c0 = 299792458.0;

template< class T >
T SbrSolver< T >::mu0 = 12.566370614E-7;

template< class T >
T SbrSolver< T >::eps0 = 8.854187817E-12;

template< class T >
T SbrSolver< T >::z0 = 376.730313451;

template< class T >
T SbrSolver< T >::pi = 3.14159265359;










#endif