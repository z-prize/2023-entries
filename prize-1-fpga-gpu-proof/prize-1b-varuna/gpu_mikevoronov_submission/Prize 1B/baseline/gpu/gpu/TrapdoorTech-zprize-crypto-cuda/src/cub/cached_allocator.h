#pragma once

struct not_my_pointer
{
  not_my_pointer(void* p) : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() { }
  virtual const char* what() const { return message.c_str(); }

private:
  std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;
  cached_allocator() { }
  ~cached_allocator() { /*free_all();*/ }

  char *allocate(std::ptrdiff_t num_bytes)
  {
	char *result = NULL;

	std::ptrdiff_t found;
	for (auto& elem : free_blocks)
	{
		if (num_bytes <= elem.first)
		{
			result = elem.second;
			found = elem.first;
			free_blocks.erase(elem.first);
			break;
		}
	}
	
	if (result == NULL)
	{
		//printf("allocate %lld\n", num_bytes);
		result = thrust::cuda::malloc<char>(num_bytes).get();
		allocated_blocks.insert(std::make_pair(result, num_bytes));
	}
	else
		allocated_blocks.insert(std::make_pair(result, found));
	
	return result;
  }

  void deallocate(char *ptr, size_t)
  {
    auto iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

public:
  void free_all()
  {
    //std::cout << "cached_allocator::free_all()" << std::endl;

	for (auto& elem : free_blocks)
	{
		//printf("free_block %lld with numSize %lld\n", elem.second, elem.first);
		thrust::cuda::free(thrust::cuda::pointer<char>(elem.second));
	}
	
	for (auto& elem : allocated_blocks)
	{
		//printf("allocated_block %lld, with numSize %lld\n", elem.first, elem.second);
		thrust::cuda::free(thrust::cuda::pointer<char>(elem.first));
	}
	
	free_blocks.clear();
	allocated_blocks.clear();
  }
};
