#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <math.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinfer1;


void xywh2xyxy(std::vector<Yolo::Detection>& res){
    for (std::size_t i = 0; i < res.size(); i++){
        float box[4];
        for (int j=0; j < 4; j++){
            memcpy(&box[j], &res[i].bbox[j], sizeof(float));
        }
        res[i].bbox[0] = box[0] - box[2] / 2.0f; // top left x
        res[i].bbox[1] = box[1] - box[3] / 2.0f; // top left y
        res[i].bbox[2] = box[0] + box[2] / 2.0f; // bottom right x
        res[i].bbox[3] = box[1] + box[3] / 2.0f; // bottom right y
    }
}


void xyxy2xywh(std::vector<Yolo::Detection>& res){
    for (std::size_t i = 0; i < res.size(); i++){
        float box[4];
        memcpy(&box, &res[i].bbox[0], 4 * sizeof(float));
        res[i].bbox[0] = box[0] + (box[2] - box[0]) / 2.0f;
        res[i].bbox[1] = box[1] + (box[3] - box[1]) / 2.0f;
        res[i].bbox[2] = (box[2] - box[0]) / 2.0f;
        res[i].bbox[3] = (box[3] - box[1]) / 2.0f;
    }
}


void clamp(float& x, int min, int max){
    if (x < min) {
        x = min;
    }
    if (x > max) {
        x = max;
    }
}


void scale_coords(std::vector<Yolo::Detection>& res, int img_w, int img_h){
    float gain = (std::min)(Yolo::INPUT_W / float(img_w), Yolo::INPUT_H / float(img_h)); // gain  = old / new
    float pad[2];
    pad[0] = (Yolo::INPUT_W - img_w * gain) / 2;
    pad[1] = (Yolo::INPUT_H - img_h * gain) / 2; // wh padding
    for (std::size_t i = 0; i < res.size(); i++){
        float box[4];
        for (int j = 0; j < 4; j++){
            memcpy(&box[j], &res[i].bbox[j], sizeof(float));
        } 
        res[i].bbox[0] = (box[0] - pad[0]) / gain; // x paddingimg_w
        res[i].bbox[1] = (box[1] - pad[1]) / gain; // y padding
        res[i].bbox[2] = (box[2] - pad[0]) / gain; // x padding
        res[i].bbox[3] = (box[3] - pad[1]) / gain; // y padding
        clamp(res[i].bbox[0], 0, img_w);
        clamp(res[i].bbox[1], 0, img_h);
        clamp(res[i].bbox[2], 0, img_w);
        clamp(res[i].bbox[3], 0, img_h);
    }
}


float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float); // 4+1+1
    std::map<float, std::vector<Yolo::Detection>> m;
    
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    // IScaleLayer: output=(input??scale+shift)^power,??????power=1
    // INetworkDefinition::addScale???kCHANNEL????????????????????????????????????
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    // INetworkDefinition::addConvolutionNd(??????tensor??????????????????????????????????????????????????????????????????)
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    // INetworkDefinition::addActivation(??????tensor???????????????)
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    // INetworkDefinition::addElementWise(tensor1???tensor2?????????)???kPROD????????????????????????
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname) {
    // nvinfer1::INetworkDefinition::addSlice(??????tensor?????????????????????????????????): ??????
    // nvinfer1::ISliceLayer
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    // ISliceLayer::getOutput?????????????????????????????????????????????
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    // nvinfer1::INetworkDefinition::addConcatenation(ITensor, ??????tensor??????)???
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    // INetworkDefinition::addPoolingNd(??????tensor?????????????????????????????????)
    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap)
{
    std::vector<float> anchors_yolo;
    int achors_nums = 24; // anchor??????=layer_nums*(3*2)
    int layer_nums = 4; // head??????
    Weights Yolo_Anchors = weightMap["model.33.anchor_grid"];
    assert(Yolo_Anchors.count == achors_nums);
    int each_yololayer_anchorsnum = Yolo_Anchors.count / layer_nums;
    const float* tempAnchors = (const float*)(Yolo_Anchors.values);
    for (int i = 0; i < Yolo_Anchors.count; i++)
    {
        if (i < each_yololayer_anchorsnum) // 6
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if ((i >= each_yololayer_anchorsnum) && (i < (2 * each_yololayer_anchorsnum)))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if ((i >= (2 * each_yololayer_anchorsnum)) && (i < (3 * each_yololayer_anchorsnum)))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
        if (i >= (3 * each_yololayer_anchorsnum))
        {
            anchors_yolo.push_back(const_cast<float*>(tempAnchors)[i]);
        }
    }
    return anchors_yolo;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, IConvolutionLayer* det_s, 
    IConvolutionLayer* det_m, IConvolutionLayer* det_l, IConvolutionLayer* det_xl)
{
    // ??????getPluginRegistry????????????TensorRT?????????creator???IPluginCreator??????
    //                                                  (pluginName, pluginVersion)
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    std::vector<float> anchors_yolo = getAnchors(weightMap);
    // ?????????????????????????????????????????????????????????[name, data, type, length]
    PluginField pluginMultidata[5];
    int NetData[4];
    NetData[0] = Yolo::CLASS_NUM;
    NetData[1] = Yolo::INPUT_W;
    NetData[2] = Yolo::INPUT_H;
    NetData[3] = Yolo::MAX_OUTPUT_BBOX_COUNT;
    pluginMultidata[0].data = NetData;
    pluginMultidata[0].length = 4; // data?????????
    pluginMultidata[0].name = "netdata";
    pluginMultidata[0].type = PluginFieldType::kFLOAT32;
    int scale[4] = { 8, 16, 32, 64 };
    int plugindata[4][8]; // 4???det?????????det[w, h, 6*anchor]
    std::string names[4];
    for (int k = 1; k < 5; k++)
    {
        plugindata[k - 1][0] = Yolo::INPUT_W / scale[k - 1];
        plugindata[k - 1][1] = Yolo::INPUT_H / scale[k - 1];
        for (int i = 2; i < 8; i++)
        {
            plugindata[k - 1][i] = int(anchors_yolo[(k - 1) * 6 + i - 2]);
        }
        pluginMultidata[k].data = plugindata[k - 1];
        pluginMultidata[k].length = 8; // data?????????
        names[k - 1] = "yolodata" + std::to_string(k);
        pluginMultidata[k].name = names[k - 1].c_str();
        pluginMultidata[k].type = PluginFieldType::kFLOAT32;
    }
    PluginFieldCollection pluginData;
    pluginData.nbFields = 5; // ????????????+4???head
    pluginData.fields = pluginMultidata;
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", &pluginData);
    ITensor* inputTensors_yolo[] = { det_xl->getOutput(0), det_l->getOutput(0), det_m->getOutput(0), det_s->getOutput(0) };
    auto yolo = network->addPluginV2(inputTensors_yolo, 4, *pluginObj);
    // printf("net")
    return yolo;
}
#endif

