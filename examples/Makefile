default:
	make default -C mnist_test
	make default -C mnist_test_recurrent
	make default -C mnist_test_convolutional

atlas:
	make atlas -C mnist_test
	make atlas -C mnist_test_recurrent
	make atlas -C mnist_test_convolutional


travis:
	make travis -C mnist_test
	make travis -C mnist_test_recurrent
	make travis -C mnist_test_convolutional

cuda:
	make cuda -C mnist_test
	make cuda -C mnist_test_recurrent
	make cuda -C mnist_test_convolutional

cuda_atlas:
	make cuda_atlas -C mnist_test
	make cuda_atlas -C mnist_test_recurrent
	make cuda_atlas -C mnist_test_convolutional


clean:
	make clean -C mnist_test
	make clean -C mnist_test_recurrent
	make clean -C mnist_test_convolutional

clang:
	make clang -C mnist_test
	make clang -C mnist_test_recurrent
	make clang -C mnist_test_convolutional

run:
	cd mnist_test && ./mnist_test_exec;
	cd mnist_test_recurrent && ./mnist_test_exec;
	cd mnist_test_convolutional && ./mnist_test_exec;

run_cuda:
	cd mnist_test && ./mnist_test_exec_cu;
	cd mnist_test_recurrent && ./mnist_test_exec_cu;
	cd mnist_test_convolutional && ./mnist_test_exec_cu;
	