#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {

        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        ////寻找最小块内存
        auto mintarget = free_block.end();
        for(auto it =free_block.begin(); it!=free_block.end(); ++it){
            if(it->second >= size){
                if(mintarget == free_block.end() || it->second < mintarget->second){
                mintarget = it;
                }
            }
        }
        ////如果有满足size的内存块进行分配
        if(mintarget != free_block.end()){
            size_t addr = mintarget->first;
            size_t offset = mintarget->second;
            if(offset == size) free_block.erase(mintarget);
            else{
                size_t new_addr = addr + size;
                size_t new_offset = offset -size;
                free_block.erase(mintarget);
                free_block.emplace(new_addr,new_offset);
            }
            used += size;//used??? peak 代表什么
            peak = std::max(peak,addr + size);
            return addr;
        }

        ////没有满足size的内存块，对最后一块内存块进行扩容
        if(!free_block.empty()){
            auto last_block = free_block.rbegin();
            size_t start = last_block->first;
            size_t offset = last_block->second;
            if(start + offset == peak){
                size_t addr = start;
                if(offset == size){
                    free_block.erase(std::prev(last_block.base()));
                    used += size;
                    peak = std::max(peak,addr + size);
                    return addr;
                }
                else{
                    free_block.erase(std::prev(last_block.base()));
                    used += size;
                    peak = std::max(peak,addr + size);
                    return addr;

                }
            }
         }

        const auto addr = peak;
        used += size;
        peak += size;
        return addr;

    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        auto result = free_block.emplace(addr , size);
        if(!result.second){
            result.first->second += size;
        }
       
        auto it = result.first;
        if(it != free_block.begin()){
            auto prev_it = std::prev(it);
            if(prev_it->first + prev_it->second == it->first){
                
                prev_it->second += it->second;
                free_block.erase(it);
                it = prev_it;
            }
        }

        auto next_it = std::next(it);
        if(next_it != free_block.end()){
            if(it->first + it->second == next_it->first){
             
                it->second += next_it->second;
                free_block.erase(next_it);
            }
        }

  
        if(addr + size == peak){
            peak = addr;
        }


        used -= size;



    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
