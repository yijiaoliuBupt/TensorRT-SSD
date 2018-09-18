#ifndef PLUGIN_FACTORY_H
#define PLUGIN_FACTORY_H


#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <map>
#include <memory>

#include "math_functions.h"

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;



class PreluPlugin : public IPlugin
{
public:
	PreluPlugin(const Weights *weights, int nbWeights);
	PreluPlugin(const void* buffer, size_t size);
	~PreluPlugin();
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims);
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream);
	int getNbOutputs() const override
    {
        return 1;
    };
    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
    void serialize(void* buffer) override;
    size_t getSerializationSize() override;
    inline size_t getWorkspaceSize(int) const override { return 0; }
    int initialize() override;
    void terminate() override;
protected:
	int input_c;
	int input_h;
	int input_w;
	int input_count;
	bool channel_shared_ {false};
	Weights mWeights;
	void* mDeviceKernel{nullptr};

private:
	void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        cudaMalloc(&deviceData, count);
        cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
        return deviceData;
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    size_t type2size(DataType type) { return sizeof(float); }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        memcpy(buffer, weights.values, weights.count * type2size(weights.type));
        buffer += weights.count * type2size(weights.type);
    }
};

template<int OutC>
class SliceLayer : public IPlugin
{
public:
    SliceLayer(){}
    SliceLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return OutC; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(3 == inputs[0].nbDims);
        return DimsCHW(1, inputs[0].d[1], inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[1],inputs[0]+1*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[2],inputs[0]+2*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[3],inputs[0]+3*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};

class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];

        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};

template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(inputs[0].d[0] * inputs[0].d[1] / OutC,OutC,inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};

class PluginFactory: public nvinfer1::IPluginFactory,
                      public nvcaffeparser1::IPluginFactory {
public:
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override  ;
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override ; 
    bool isPlugin(const char* name) override;
    void destroyPlugin();
private:
    std::map<std::string, plugin::INvPlugin*> _nvPlugins; 
};

#endif