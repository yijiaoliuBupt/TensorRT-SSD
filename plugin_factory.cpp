#include "plugin_factory.h"
#include "util/math_functions.h"
#include <iostream>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

PreluPlugin::PreluPlugin(const Weights *weights, int nbWeights){
    assert(nbWeights==1);
    mWeights = weights[0];
    assert(mWeights.type == DataType::kFLOAT || mWeights.type == DataType::kHALF);
    mWeights.values = malloc(mWeights.count*type2size(mWeights.type));
    memcpy(const_cast<void*>(mWeights.values),weights[0].values,mWeights.count*type2size(mWeights.type));
}

PreluPlugin::PreluPlugin(const void* buffer, size_t size)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    read<int>(d,input_c);
    read<int>(d,input_h);
    read<int>(d,input_w);
    read<int>(d,input_count);
    read<bool>(d,channel_shared_);
    read<int64_t>(d,mWeights.count);
    read<DataType>(d,mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));//deserializeToDevice(d,mDeviceKernel,mWeights.count);
    memcpy(const_cast<void*>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d += mWeights.count * type2size(mWeights.type);
    assert(d == a + size);
}

PreluPlugin::~PreluPlugin()
{   

    //std::cout << "~PreluPlugin  "<< mWeights.values << std::endl;
    if (mWeights.values){
        free(const_cast<void*>(mWeights.values));
    }
}

Dims PreluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}


void PreluPlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int){
    input_c = inputs[0].d[0]; 
    input_h = inputs[0].d[1];
    input_w = inputs[0].d[2];
    input_count = input_c * input_h * input_w;
}

size_t PreluPlugin::getSerializationSize() {
    return 4*sizeof(int) + sizeof(bool) + sizeof(mWeights.count) 
    + sizeof(mWeights.type) +  mWeights.count * type2size(mWeights.type);
}

void PreluPlugin::serialize(void* buffer) {
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, input_c);
    write(d, input_h);
    write(d, input_w);
    write(d, input_count);
    write(d, channel_shared_);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d,mWeights);
    assert(d == a + getSerializationSize());
}

int PreluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    const float *bottom_data = reinterpret_cast<const float*>(inputs[0]);
    float *top_data = reinterpret_cast<float*>(outputs[0]);

    const int count = batchSize * input_count;
    const int dim = input_h*input_w;
    const int channels = input_c;
    const int div_factor = channel_shared_ ? channels : 1; //channel_shared_ default is false

    PReLUForward(count,channels,dim,bottom_data,top_data,mDeviceKernel,div_factor);

    return 0;
}

int PreluPlugin::initialize(){
    //std::cout << "~initialize  "<< mDeviceKernel << std::endl;
    cudaMalloc(&mDeviceKernel,mWeights.count*type2size(mWeights.type));
    cudaMemcpy(mDeviceKernel,mWeights.values,mWeights.count*type2size(mWeights.type),cudaMemcpyHostToDevice);
    return 0;
}

void PreluPlugin::terminate(){
    if (mDeviceKernel){
        //std::cout << "~terminate  "<< mDeviceKernel << std::endl;
        cudaFree(mDeviceKernel);
        mDeviceKernel = nullptr;
    }
}


bool PluginFactory::isPlugin(const char* name)
{
    return ( !strcmp(name,"conv3_3_norm_mbox_loc_perm")
            || !strcmp(name,"conv3_3_norm_mbox_loc_flat")
            || !strcmp(name,"conv3_3_norm_mbox_conf_slice")
            || !strcmp(name,"conv3_3_norm_mbox_conf_out")
            || !strcmp(name,"conv3_3_norm_mbox_conf_perm")
            || !strcmp(name,"conv3_3_norm_mbox_conf_flat")
            || !strcmp(name,"conv3_3_norm_mbox_priorbox")
           ||!strcmp(name,"conv4_3_norm_mbox_loc_perm")
           || !strcmp(name,"conv4_3_norm_mbox_conf_perm")
           || !strcmp(name,"conv4_3_norm_mbox_priorbox")
           || !strcmp(name,"conv5_3_norm_mbox_loc_perm")
           || !strcmp(name,"conv5_3_norm_mbox_conf_perm")
           || !strcmp(name,"conv5_3_norm_mbox_priorbox")
           || !strcmp(name,"fc7_mbox_loc_perm")
           || !strcmp(name,"fc7_mbox_conf_perm")
           || !strcmp(name,"fc7_mbox_priorbox")
           || !strcmp(name,"mbox_conf_reshape")
           || !strcmp(name,"mbox_loc")
           || !strcmp(name,"mbox_conf")
           || !strcmp(name,"mbox_priorbox")
           || !strcmp(name,"conv4_3_norm_mbox_loc_flat")
           || !strcmp(name,"conv4_3_norm_mbox_conf_flat")
           || !strcmp(name,"conv5_3_norm_mbox_loc_flat")
           || !strcmp(name,"conv5_3_norm_mbox_conf_flat")
           || !strcmp(name,"fc7_mbox_loc_flat")
           || !strcmp(name,"fc7_mbox_conf_flat")
           || !strcmp(name,"mbox_conf_flatten")
           || !strcmp(name,"detection_out")
	       );
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    assert(isPlugin(layerName));

    if(! strcmp(layerName,"conv4_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv4_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"fc7_mbox_loc_perm")
             || !strcmp(layerName,"fc7_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_loc_perm")
    )
    {
        _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple({0,2,3,1}));  
        return _nvPlugins.at(layerName);
    }
    else if (!strcmp(layerName,"conv3_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {16.0f};   
         
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
         
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 8.0f;  
        params.stepW = 8.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
        return _nvPlugins.at(layerName);
    }
    else if(! strcmp(layerName,"conv4_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {32.0f};   
           
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
          
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 8.0f;  
        params.stepW = 8.0f;  
        params.offset = 0.5f;  

        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"conv5_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {64.0f};   
           
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
        
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 16.0f;  
        params.stepW = 16.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if( !strcmp(layerName,"fc7_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {128.0f};   
        //float maxSize[1] = {60.0f};   
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
        //params.maxSize = (float*)maxSize;  
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 32.0f;  
        params.stepW = 32.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"detection_out")){
        plugin::DetectionOutputParameters params {true, false, 0, 2, 200, 200, 0.05, 0.3, plugin::CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true};
        _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(params);  
        return _nvPlugins.at(layerName); 
    }
    else if(!strcmp(layerName,"mbox_conf_reshape")){
    	assert(nbWeights == 0 && weights == nullptr);
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new Reshape<2>());
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_loc")
    	     || !strcmp(layerName,"mbox_conf")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_out")){
    	_nvPlugins[layerName] = plugin::createConcatPlugin(1,false);
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_priorbox")){
        _nvPlugins[layerName] = plugin::createConcatPlugin(2,false);
        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv4_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"conv3_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"conv3_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"fc7_mbox_loc_flat")
           || !strcmp(layerName,"fc7_mbox_conf_flat")
           || !strcmp(layerName,"mbox_conf_flatten")
           )
    {
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());
    	return _nvPlugins.at(layerName);
    }
    else if (!strcmp(layerName,"conv3_3_norm_mbox_conf_slice")){
        _nvPlugins[layerName] = (plugin::INvPlugin*)(new SliceLayer<4>());
        return _nvPlugins.at(layerName);
    }
    else{  
        std::cout << "error: " << layerName <<"doesn't implement"<< std::endl;
        assert(0);  
        return nullptr;  
    }  
}


IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    std::cout<<layerName<<std::endl;
	assert(isPlugin(layerName));

    if(!strcmp(layerName,"conv4_3_norm")
        || !strcmp(layerName,"conv5_3_norm")
        ){
        _nvPlugins[layerName] = plugin::createSSDNormalizePlugin(serialData,serialLength);  

        return _nvPlugins.at(layerName);
    }else if(! strcmp(layerName,"conv4_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv4_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"fc7_mbox_loc_perm")
             || !strcmp(layerName,"fc7_mbox_conf_perm")
             ){
        _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
	}else if(! strcmp(layerName,"conv4_3_norm_mbox_priorbox")
            || !strcmp(layerName,"conv5_3_norm_mbox_priorbox")
            || !strcmp(layerName,"fc7_mbox_priorbox")
            ){
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }else if(! strcmp(layerName,"detection_out")){
        _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }else{  
         assert(0);  
         return nullptr;  
    }  
}


void PluginFactory::destroyPlugin()  
{  
    for (auto it=_nvPlugins.begin(); it!=_nvPlugins.end(); ++it){  
        std::cout<<it->first<<std::endl;
        it->second->destroy();  
        _nvPlugins.erase(it);  
    } 
}

