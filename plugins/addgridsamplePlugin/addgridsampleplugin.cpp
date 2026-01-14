#include "addgridsampleplugin.hpp"
#include <cassert>
#include <mutex>
#include <string>
#include "addgridsample.cuh"

using nvinfer1::plugin::AddGridSamplePlugin;
using nvinfer1::plugin::AddGridSamplePluginCreator;
// #define PLUGIN_ASSERT(assertion)                                                                                       \
//     do                                                                                                                 \
//     {                                                                                                                  \
//         if (!(assertion))                                                                                              \
//         {                                                                                                              \
//                                                      \
//         }                                                                                                              \
//     } while (0)
REGISTER_TENSORRT_PLUGIN(AddGridSamplePluginCreator);
namespace
{
constexpr char const* ADD_GRID_SAMPLE_VERSION{"1"};
constexpr char const* ADD_GRID_SAMPLE_NAME{"MyGridSample"};
}

template <typename scalar_t>
void writeToBuffer(char*& buffer, const scalar_t& val)
{
    *reinterpret_cast<scalar_t*>(buffer) = val;
    buffer += sizeof(scalar_t);
}

template <typename scalar_t>
scalar_t readFromBuffer(const char*& buffer)
{
    scalar_t val = *reinterpret_cast<const scalar_t*>(buffer);
    buffer += sizeof(scalar_t);
    return val;
}

AddGridSamplePlugin::AddGridSamplePlugin(const std::string name,
                                        size_t input_channel,
                                        size_t input_depth,
                                        size_t input_height, 
                                        size_t input_width,
                                        size_t grid_depth,
                                        size_t grid_height,
                                        size_t grid_width,
                                        bool aligncorners,
                                        std::string interpolation_mode,
                                        std::string padding_mode,
                                        DataType data_type):
    m_layername(name),
    m_input_channel(input_channel),
    m_input_depth(input_depth),
    m_input_height(input_height),
    m_input_width(input_width),
    m_grid_depth(grid_depth),
    m_grid_height(grid_height),
    m_grid_width(grid_width),
    m_align_corners(aligncorners),
    m_mode(interpolation_mode),
    m_padding_mode(padding_mode),
    m_data_type(data_type){}
AddGridSamplePlugin::AddGridSamplePlugin(const std::string name,
                                        bool aligncorners,
                                        std::string interpolation_mode,
                                        std::string padding_mode):
    m_layername(name),
    m_align_corners(aligncorners),
    m_mode(interpolation_mode),
    m_padding_mode(padding_mode){}


AddGridSamplePlugin::~AddGridSamplePlugin() {}

/***************** IPluginV3 Methods *****************/ 
nvinfer1::IPluginCapability* AddGridSamplePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        assert(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        std::cerr << "Plugin -> AddGridSamplePlugin::getCapabilityInterface() error : " << e.what() << std::endl;
    }
    return nullptr;
}
AddGridSamplePlugin* AddGridSamplePlugin::clone() noexcept{
        auto plugin = new AddGridSamplePlugin(m_layername, 
                                         m_input_channel, 
                                         m_input_depth, 
                                         m_input_height, 
                                         m_input_width, 
                                         m_grid_depth, 
                                         m_grid_height, 
                                         m_grid_width, 
                                         m_align_corners, 
                                         m_mode, 
                                         m_padding_mode, 
                                         m_data_type);
    plugin->setPluginNamespace(m_namespace.c_str());   
    return plugin;    
}



/***************** IPluginV3OneBuild(V2) Methods *****************/
int32_t AddGridSamplePlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, 
                                             DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept{
    assert(nbInputs == m_in_num && nbOutputs == m_out_num);
    // for 3d grid sample, the input should be 5 dims
    assert(in[0].desc.dims.nbDims == 5);        
    assert(in[1].desc.dims.nbDims == 5);

    m_batch = in[0].desc.dims.d[0];
    m_input_channel = in[0].desc.dims.d[1];
    m_input_depth = in[0].desc.dims.d[2];
    m_input_height = in[0].desc.dims.d[3];
    m_input_width = in[0].desc.dims.d[4];

    m_grid_depth = in[1].desc.dims.d[1];
    m_grid_height = in[1].desc.dims.d[2];
    m_grid_width = in[1].desc.dims.d[3];
    m_data_type = in[0].desc.type;
    assert(m_batch == in[1].desc.dims.d[0] && m_input_depth==m_grid_depth && m_input_height==m_grid_height && m_input_width==m_grid_width);
    assert(in[1].desc.dims.d[4] == 3);
    return 0;
}
int32_t AddGridSamplePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept{
    assert(inputTypes != nullptr);
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}
int32_t AddGridSamplePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept{
    assert(inputs != nullptr);
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
    outputs[0] = inputs[0];

    
    return 0;
}
bool AddGridSamplePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept{
    assert(nbInputs == 2 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
    bool condition = inOut[pos].desc.format == TensorFormat::kLINEAR;
    bool condition_1 = inOut[pos].desc.type == DataType::kFLOAT || inOut[pos].desc.type == DataType::kHALF;
    bool condition_2 = inOut[pos].desc.type == inOut[0].desc.type;
    return condition && condition_1 && condition_2;   
}
int32_t AddGridSamplePlugin::getNbOutputs() const noexcept{
    return m_out_num;
}
size_t AddGridSamplePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept{
        return 0;
}

/***************** IPluginV3OneCore Methods *****************/
char const* AddGridSamplePlugin::getPluginName() const noexcept{
    return ADD_GRID_SAMPLE_NAME;
}
char const* AddGridSamplePlugin::getPluginVersion() const noexcept{
    return ADD_GRID_SAMPLE_VERSION;
}
char const* AddGridSamplePlugin::getPluginNamespace() const noexcept{
    return m_namespace.c_str();
}
/***************** IPluginV3OneRuntime Methods *****************/
int32_t AddGridSamplePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept{
    assert(in != nullptr);
    assert(out != nullptr);
    assert(nbInputs == m_in_num);
    assert(nbOutputs == m_out_num);
    return 0;
}

int32_t AddGridSamplePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept{
    int status = -1;
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error in enqueue: %s\n", cudaGetErrorString(err));
    }
    nvinfer1::Dims input0_dims = inputDesc[0].dims;
    m_batch = input0_dims.d[0];
    m_input_channel =  input0_dims.d[1];
    m_grid_depth = m_input_depth =  input0_dims.d[2];
    m_grid_height = m_input_height =  input0_dims.d[3];
    m_grid_width = m_input_width =  input0_dims.d[4];
    if(m_data_type == DataType::kFLOAT) {
        status = grid_sample_3d_cuda<float>(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            m_batch, m_input_channel, m_input_depth, m_input_height, m_input_width,
            m_grid_depth, m_grid_height, m_grid_width,
            m_align_corners,
            m_mode,
            m_padding_mode,
            static_cast<float*>(outputs[0]),
            stream
        );
    } else if(m_data_type == DataType::kHALF) {
        status = grid_sample_3d_cuda<half>(
            static_cast<const half*>(inputs[0]),
            static_cast<const half*>(inputs[1]),
            m_batch, m_input_channel, m_input_depth, m_input_height, m_input_width,
            m_grid_depth, m_grid_height, m_grid_width,
            m_align_corners,
            m_mode,
            m_padding_mode,
            static_cast<half*>(outputs[0]),
            stream
        );
    }

    return status;
}
nvinfer1::IPluginV3* AddGridSamplePlugin::attachToContext(IPluginResourceContext* context) noexcept{
    auto* newPlugin = clone();
    return newPlugin;
}
nvinfer1::PluginFieldCollection const* AddGridSamplePlugin::getFieldsToSerialize() noexcept{
    m_data_to_serialize.clear();

    const char* mode_cstr = m_mode.c_str();
    const char* padding_mode_cstr = m_padding_mode.c_str();
    m_data_to_serialize.emplace_back("mode", mode_cstr, PluginFieldType::kCHAR, m_mode.length()+1);
    m_data_to_serialize.emplace_back("padding_mode", padding_mode_cstr, PluginFieldType::kCHAR, m_padding_mode.length()+1);
    m_data_to_serialize.emplace_back("align_corners", &m_align_corners, PluginFieldType::kINT32, 1);

    m_FC_to_serialize.nbFields = m_data_to_serialize.size();
    m_FC_to_serialize.fields = m_data_to_serialize.data();
    return &m_FC_to_serialize;
}
void AddGridSamplePlugin::setPluginNamespace(char const* pluginNamespace) noexcept{
    m_namespace = pluginNamespace;
}



AddGridSamplePluginCreator::AddGridSamplePluginCreator(){
    static std::mutex sMutex;
    std::lock_guard<std::mutex> guard(sMutex);
    m_plugin_attributes.clear();
    m_plugin_attributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kCHAR, 0));
    m_plugin_attributes.emplace_back(PluginField("padding_mode", nullptr, PluginFieldType::kCHAR, 0));
    m_plugin_attributes.emplace_back(PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

    m_field_collection.nbFields = m_plugin_attributes.size();
    m_field_collection.fields = m_plugin_attributes.data();
}
nvinfer1::IPluginV3* AddGridSamplePluginCreator::createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept{
    const PluginField* fields = fc -> fields;
    int nbFields = fc -> nbFields;
    std::string mode = "bilinear";      // 注意：这里应该是mode，不是interpolationMode
    std::string padding_mode = "zeros";
    int align_corners = 0;

    for(int i = 0; i < nbFields; i++) {
        const char* field_name = fields[i].name;
        const void* field_data = fields[i].data;

        if(!strcmp(field_name, "mode")) {
            mode = std::string(reinterpret_cast<const char*>(field_data));
        }

        if(!strcmp(field_name, "padding_mode")) {
            padding_mode = std::string(reinterpret_cast<const char*>(field_data));
        }

        if(!strcmp(field_name, "align_corners")) {
            align_corners = *(reinterpret_cast<const int*>(field_data));
        }
        
    }
    auto plugin = new AddGridSamplePlugin(name, align_corners,  mode, padding_mode);
    plugin->setPluginNamespace(m_namespace.c_str());
    return plugin;
}
nvinfer1::PluginFieldCollection const* AddGridSamplePluginCreator::getFieldNames() noexcept{
    return &m_field_collection;
}
char const* AddGridSamplePluginCreator::getPluginName() const noexcept{
    return ADD_GRID_SAMPLE_NAME;
}
char const* AddGridSamplePluginCreator::getPluginVersion() const noexcept{
    return ADD_GRID_SAMPLE_VERSION;
}
char const* AddGridSamplePluginCreator::getPluginNamespace() const noexcept{
    return m_namespace.c_str();
}
void AddGridSamplePluginCreator::setPluginNamespace(char const* libNamespace) noexcept{
    if(libNamespace!=nullptr){
        m_namespace = libNamespace;
    }else{
        assert(false);
    }
}