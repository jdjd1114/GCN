
#if !defined(_ERROR_UTIL_H_)
#define _ERROR_UTIL_H_

#include<sstream>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>

#define FatalError(s) {							\
	std::stringstream _where, _message;				\
	_where << __FILE__ << ':' << __LINE__;				\
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
	std::cerr << _message.str() << "\nAborting...\n";                \
	cudaDeviceReset();                                                 \
	exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
	std::stringstream _error;                                          \
	if (status != CUDNN_STATUS_SUCCESS) {                              \
		_error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
		FatalError(_error.str());                                        \
	}                                                                  \
}

#define checkCudaErrors(status) {                                      \
	std::stringstream _error;                                          \
	if (status != 0) {                                                 \
		_error << "Cuda failure\nError: " << cudaGetErrorString(status); \
		FatalError(_error.str());                                        \
	}                                                                  \
}

#define checkCublasErrors(status) {                                    \
	std::stringstream _error;                                          \
	if (status != 0) {                                                 \
		_error << "Cublas failure\nError code " << status;        \
		FatalError(_error.str());                                        \
	}                                                                  \
}

#endif // _ERROR_UTIL_H_
