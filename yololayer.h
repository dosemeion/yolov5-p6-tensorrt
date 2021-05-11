#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3; // 每个尺度anchor个数
    static constexpr float IGNORE_THRESH = 0.45f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 6;
    static constexpr int INPUT_H = 384;
    static constexpr int INPUT_W = 640;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        // 直接解析网络时候需要用到
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        
        // 反序列化时候需要用到
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();
        
        // 直接return输出节点数
        int getNbOutputs() const override
        {
            return 1;
        }
        
        // return输出的维度信息，如：return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2])
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
        
        // 调用enqueue的时候需要用到的资源先在这里Initialize，这个函数是在engine创建之后enqueue调用之前调用的，不需要Initialize则直接 return 0;
        int initialize() override;
        
        // 释放Initialize申请的资源，在enqueue调用之后且engine销毁之后调用
        virtual void terminate() override {};
        
        // 设置工作空间，不需要直接 return 0;
        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
        
        // 前向计算的核心函数，计算逻辑在这里实现，可以使用cublas实现或者自己写cuda核函数实现
        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
        
        // 在这里返回正确的序列化数据的长度，如我要序列化数据类型和数据维度：return sizeof(data_type) + sizeof(chw);
        virtual size_t getSerializationSize() const override;

         // 序列化函数，在这里把反序列化时需要用到的参数或数据序列化
        virtual void serialize(void* buffer) const override;

        // pos索引到的input/output的数据格式（format）和数据类型（datatype）如果都支持则返回true
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        // 返回自定义类型
        const char* getPluginType() const override;

        // 返回plugin version
        const char* getPluginVersion() const override;

        // 销毁对象
        void destroy() override;

        // 在这里new一个该自定义类型并返回
        IPluginV2IOExt* clone() const override;

        // 设置命名空间，用来在网络中查找和创建plugin
        void setPluginNamespace(const char* pluginNamespace) override;

         // 返回plugin对象的命名空间
        const char* getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

        void detachFromContext() override;

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount; // 预测尺度数量
        int mClassCount; // 类别数
        int mYoloV5NetWidth; // 网络输入宽度
        int mYoloV5NetHeight; // 网络输入高度
        int mMaxOutObject; // 最大检测数量
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const PluginFieldCollection* getFieldNames() override;

        // 创建自定义层plugin的对象并返回
        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 

