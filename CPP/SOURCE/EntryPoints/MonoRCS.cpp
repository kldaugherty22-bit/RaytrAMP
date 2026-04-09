#include "TypeDef.hpp"
#include "Triangle.hpp"
#include "TriangleMesh.hpp"
#include "MortonManager.hpp"
#include "BvhGenerator.hpp"
#include "BvhNodeTypes.hpp"
#include "ReducedBvhArray.hpp"
#include "RayPool.hpp"
#include "DepthMapGenerator.hpp"
#include "SbrSolver.hpp"

#include "TypeDef.hpp"
#include "Triangle.hpp"
#include "TriangleMesh.hpp"
#include "MortonManager.hpp"
#include "BvhGenerator.hpp"
#include "BvhNodeTypes.hpp"
#include "ReducedBvhArray.hpp"
#include "RayPool.hpp"
#include "DepthMapGenerator.hpp"
#include "SbrSolver.hpp"

#include <iostream>
#include <exception>
#include <cstdio>

/*
	Args (3 total):
	[0]: executable name
	[1]: input .rba file name
	[1]: input .obs file name
	[2]: output .rcs file name
*/
int main( int argc, char *argv[] )
{
	std::setvbuf(stdout, NULL, _IONBF, 0);

	try
	{
		std::cout << "MonoRCS start\n";

		auto tStart = Clock::now();

		if( argc != 4 )
		{
			std::cerr << "Command line arg count must be 3!\n";
			return 1;
		}

		std::string rbaFilePath = std::string( argv[1] );
		std::string obsFilePath = std::string( argv[2] );
		std::string rcsFilePath = std::string( argv[3] );

		std::cout << "Loading RBA: " << rbaFilePath << "\n";
		ReducedBvhArray< Float > reducedBvhArray;
		reducedBvhArray.Load( rbaFilePath );
		std::cout << "Loaded RBA\n";

		std::cout << "Loading OBS: " << obsFilePath << "\n";
		ObservationArray< Float > observationArray;
		observationArray.Load( obsFilePath );
		std::cout << "Loaded OBS\n";

		std::cout << "Creating RCS array: obsCount = " << observationArray.obsCount_ << "\n";
		RcsArray< Float > rcsArray;
		rcsArray.Initialize( observationArray.obsCount_ );
		std::cout << "Created RCS array\n";

		std::cout << "Calling MonostaticRcsGpu\n";
		SbrSolver< Float > sbrSolver;
		sbrSolver.MonostaticRcsGpu( reducedBvhArray, observationArray, rcsArray );
		std::cout << "Returned from MonostaticRcsGpu\n";

		std::cout << "Saving RCS file: " << rcsFilePath << "\n";
		rcsArray.Save( rcsFilePath );
		std::cout << "Saved RCS file\n";

		auto tTotal = std::chrono::duration_cast< std::chrono::milliseconds >( Clock::now() - tStart ).count();
		std::cout << "Finished in " << tTotal << " ms\n";

		return 0;
	}
	catch( const std::exception& e )
	{
		std::cout << "std::exception: " << e.what() << "\n";
		return 1;
	}
	catch( ... )
	{
		std::cout << "Unknown exception\n";
		return 1;
	}
}