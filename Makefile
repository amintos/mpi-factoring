mpi: mpisvd/main.cpp mpisvd/main.h
	mpicxx mpisvd/main.cpp -o svd

tau: mpisvd/main.cpp mpisvd/main.h
	taucxx mpisvd/main.cpp -o svd

