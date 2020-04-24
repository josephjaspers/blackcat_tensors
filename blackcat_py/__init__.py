def _initialize_bc_tensors(blackcat_tensors_path="../blackcat/tensors.h"):

	blas_options = [
		'blas_mkl_info',
		'blas_opt_info',
		'blis_info',
		'openblas_info']

	get_config = lambda opt: getattr(np.__config__, opt, None)
	first = lambda iter: next(x for x in iter if x)
	blas_library_details = first(get_config(opt) for opt in blas_options)
	libs = blas_library_details['libraries']
	paths = blas_library_details['library_dirs']

	for path, library in zip(libs, paths):
		library_path = os.path.join(path,library)
		cppyy.load_library(library_path)

	cppyy.include(blackcat_tensors_path)

_initialize_bc_tensors()
_pythonized_tensor_types = {}
from cppyy.gbl import bc  # import c++ namespace
