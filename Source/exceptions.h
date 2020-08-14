#pragma once

//Cuda Vector exceptions

class outOfBounds : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Access was attempted to an out of bounds index.";
	}
};

//Shared vector exceptions

class sizeChangedException : public std::exception {
	virtual const char* what() const throw()
	{
		return "Device vector was not rendered or host vector size changed while device vector was active.";
	}
};

class locationChangeException : public std::exception {
	virtual const char* what() const throw()
	{
		return "Location has changed from host to device (or vise versa) while an allocation is active.";
	}
};

class invalidDeletionLocation : public std::exception {
	virtual const char* what() const throw() {
		return "The deletion index was either larger than the current vector or contained a deleted member.";
	}
};

class invalidTransientVector : public std::exception {
	virtual const char* what() const throw() {
		return "The given size vector was invalid for the dimensionality of the shared vector.";
	}
};

//Dataset exceptions

class noBatchException : public std::exception
{
	virtual const char* what() const throw()
	{
		return "No batch has been bound for this dataset. Please run factory getBatch before this call.";
	}
};

class invalidSplitParameter : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Parameter to split dataset must be between 0 and 1 (non-inclusive).";
	}
};

class invalidWindowParameter : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Parameters to getWindow must describe a valid location 0-1 of the dataset.";
	}
};

class classRequestAsRegression : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Attempted to resolve class output as a regression output.";
	}
};

class invalidInputID : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Requested input does not exist in dataset.";
	}
};

class invalidOutputID : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Requested output does not exist in dataset.";
	}
};

//Dataset factory exceptions
class fileNotFoundException : public std::exception
{
	virtual const char* what() const throw()
	{
		return "A file was not found on the system with the given filename and filepath.";
	}
};

class negativeNoiseParameterException : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Output noise must be non-negative.";
	}
};

class noFunctionsSelectedException : public std::exception
{
	virtual const char* what() const throw()
	{
		return "There must be at least one function selected in the input vector.";
	}
};

class invalidCSVFormatException : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Invaild input in the CSV header.";
	}
};