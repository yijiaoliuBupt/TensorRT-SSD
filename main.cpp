#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cmath>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "NvInferPlugin.h"
#include <sys/time.h>
#include <fstream>


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static Logger gLogger;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME1 = "detection_out";
const char* OUTPUT_BLOB_NAME2 = "detection_out2";


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,
					 nvcaffeparser1::IPluginFactory* pluginFactory,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
  
	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
															  modelFile.c_str(),
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1000 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);  
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, std::vector<float*> inputs,std::vector<float*> outputs, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	//assert(engine.getNbBindings() == 2);
    void* buffers[3];
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
     
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex)), 
            outputDims1 = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex1)),
            outputDims2 = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex2));


	// create GPU buffers and a stream
	cudaMalloc(&buffers[inputIndex], batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float));
	cudaMalloc(&buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float));
    cudaMalloc(&buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float));

	cudaStream_t stream;
	cudaStreamCreate(&stream);


	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	cudaMemcpyAsync(buffers[inputIndex], input, batchSize * inputDims.c() * inputDims.h() * inputDims.w()  * sizeof(float), cudaMemcpyHostToDevice, stream);
	context.enqueue(batchSize, buffers, stream, nullptr);
	cudaMemcpyAsync(LocProb, buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(PriorBoxProb, buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	cudaFree(buffers[inputIndex]);
	cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);

}


int main(int argc, char** argv)
{

	const char* model_def = argv[1];
    const char* weights_def = argv[2];
    const char* image_name = argv[3];

    PluginFactory pluginFactory;
    
    IHostMemory *gieModelStream{nullptr};
   	caffeToGIEModel(model_def,weights_def, std::vector < std::string > { OUTPUT_BLOB_NAME1,OUTPUT_BLOB_NAME2},1,&pluginFactory, gieModelStream);
    pluginFactory.destroyPlugin();
    
    std::ofstream out("engine.binary",std::ios::out|std::ios::binary);
    out.write((const char*)(gieModelStream->data()),gieModelStream->size());
    out.close();
    

    int engine_buffer_size;

    std::ifstream in("engine.binary",std::ios::in | std::ios::binary);
    in.seekg(0,std::ios::end);
    engine_buffer_size = in.tellg();
    in.seekg(0,std::ios::beg);
    std::shared_ptr<char> engine_buffer {new char[engine_buffer_size]};
    in.read(engine_buffer.get(),engine_buffer_size);
    in.close();

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(engine_buffer.get(), engine_buffer_size, &pluginFactory);
    std::cout << "RT deserialize done!" << std::endl;

    if (gieModelStream) 
    {
        gieModelStream->destroy();
        gieModelStream = nullptr;
    }

	IExecutionContext *context = engine->createExecutionContext();
    std::cout << "RT createExecutionContext done!" << std::endl;

    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), 
        outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);
     
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(inputIndex)), 
            outputDims1 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex1)),
            outputDims2 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex2));
    
	// run inference
	float *detection_out = new float[outputDims1.c() * outputDims1.h() *outputDims1.w()];
    float *detection_out_2 = new float[outputDims2.c() * outputDims2.h() *outputDims2.w()]; 
    

    cv::Mat img = cv::imread(image_name,1);
    cv::Size dsize = cv::Size(inputDims.h(),inputDims.w());
    cv::Mat imgResize;
    cv::resize(img, imgResize, dsize, 0, 0 , cv::INTER_LINEAR);

    float means[3] = {104.0, 117.0, 123.0};

    float *data = new float[inputDims.c() * inputDims.h() * inputDims.w()];

    for (int i = 0; i < imgResize.rows; ++i){
        for (int j = 0; j < imgResize.cols; ++j){
            data[0*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[0]) - means[0];
            data[1*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[1]) - means[1];
            data[2*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[2]) - means[2];
        }
    }

    doInference(*context, data, detection_out, detection_out_2, 1);

    delete[] data;
    delete[] detection_out;
    delete[] detection_out_2;

    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();

    return 0;
}
