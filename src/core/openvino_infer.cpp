#include "openvino_infer.hpp"
#include <format>
#include "inference_common.hpp"


const DEVICE_TYPE OpenVinoInfer::GetInferenceType() const{
    return DEVICE_TYPE::OpenVino;
}


void OpenVinoInfer::CreateInferenceEngine(){
    m_model = nullptr;
    m_compiled_model = nullptr;
    m_infer_request = nullptr;

    m_core = std::make_unique<ov::Core>(ov::Core());
    // 获取可用设备
    std::vector<std::string> devices = m_core->get_available_devices();

    // 检查是否有可用设备
    if (devices.empty()) {
        std::cout << "<(E`_`E)> Error: Can't find any available device to support OpenVino..." <<std::endl;
        throw std::runtime_error("No OpenVINO devices available");
    }

    m_device = "CPU";
    // 优先选择 GPU，找到了可用的GPU就设置为GPU
    if (std::find(devices.begin(), devices.end(), "GPU") != devices.end()) {
        m_device = "GPU";
    }
    //设置openvino缓存路径 存储插件缓存数据
    m_core->set_property(m_device, ov::cache_dir("OpenVinoCache_" + m_device));
    std::cout << std::format("<(*^_^*)> OpenVino[{}] Inference Created Successfully\n", m_device);
};


ResultData<std::string> OpenVinoInfer::LoadModel(std::string file_path,
                                        std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
                                        std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts){
    //记录输入输出形状
    m_input_layouts = t_input_layouts;
    m_output_layouts = t_output_layouts;

    ResultData<std::string> return_data;
    if (!m_core) {
        return_data.error_message = "<(E`_`E)> ov::Core is NULL, please execute CreateInferenceEngine().";
        return return_data;
    }
    
    InferenceCommon::TryFunction([&](){
        std::string model_path = file_path;
        if (file_path.find(".xml") == std::string::npos) {
            int idx = file_path.find_last_of('/');
            if (idx != std::string::npos) {
                idx+=1;
                file_path = file_path.substr(idx,file_path.size()-idx);
            }
            int idx_1 = file_path.find_last_of('\\');
            if (idx_1 != std::string::npos) {
                idx_1+=1;
                file_path = file_path.substr(idx_1,file_path.size()-idx_1);
            }
            std::string save_path = std::format("{}/{}.xml",MODELPATH,file_path.substr(0,file_path.find_last_of('.')));
            std::cout << "    Save model path: " << save_path << std::endl;
            return_data.result_info = save_path;
            if (!std::filesystem::exists(save_path)) {
                //临时创建一个模型用于创建引擎文件
                auto temp_model = m_core->read_model(model_path);
                ov::save_model(temp_model,save_path);
                model_path = save_path;
            }
        }
        m_model = m_core->read_model(model_path);
        std::cout << "<(*^_^*)> Inference model (OpenVino) has been created successfully."<< std::endl;;
    },return_data);
    if (return_data.error_message.empty())
        return_data.result_state = true;
    return return_data;
}

ResultData<bool> OpenVinoInfer::CreateEngine(std::string& engine_path){

    ResultData<bool> return_data;
    if (!m_model) {
        return_data.error_message = "<(E`_`E)> Please execute LoadModel() first, then createEngine().";
        return return_data;
    }
    InferenceCommon::TryFunction([&](){
        auto ppp = ov::preprocess::PrePostProcessor(m_model);

        //Input
        MY_ASSERT(m_model->inputs().size() == m_input_layouts.size(),"model->inputs().size() != m_input_layouts.size()");
        for (int input_idx = 0; input_idx < m_model->inputs().size(); input_idx++)
        {
            auto input_layout = m_input_layouts[input_idx];
            ov::PartialShape input_partial_shape = m_model->input(input_idx).get_partial_shape();
            if (input_partial_shape.size() != input_layout.second.size()) {
                return_data.error_message = "<(E`_`E)> Please check your input_layout dims";
                return;
            }
            std::cout << "<(*^_^*)> Model input Shape: ";
            for (int i = 0; i < input_partial_shape.size(); i++)
            {
                std::cout << input_partial_shape[i] << (i != input_partial_shape.size() - 1 ? " x " : "");
            }
            std::cout << "   VS    Your input Shape: ";
            std::vector<ov::Dimension> input_dims;
            for (int i = 0; i < input_layout.second.size(); i++)
            {
                std::cout << input_layout.second[i] << (i != input_layout.second.size() - 1 ? " x " : "");
                auto dim = input_layout.second[i];
                if (dim == 0 || dim == -1) {
                    input_dims.push_back(ov::Dimension::dynamic());
                    continue;
                }
                input_dims.push_back(dim);
            }
            std::cout << "   LayOut:" << input_layout.first << std::endl;
            ov::PartialShape input_shape = ov::PartialShape(input_dims);
            ppp.input(input_idx).tensor()
                .set_element_type(ov::element::Type_t::f32)
                .set_layout(ov::Layout(input_layout.first)) // OpenVINO输入格式(NCHW\NHWC\NC?\NC...\N...C)
                .set_shape(input_shape); 
            std::cout << "<(*^_^*)> Input tensor has been setted on element_type/layout/shape."<< std::endl;

            ppp.input(input_idx).model().set_layout(ov::Layout(input_layout.first));
            std::cout << "<(*^_^*)> Input model has been setted on layout."<< std::endl;;
        }

        //Output
        MY_ASSERT(m_model->get_output_size() == m_output_layouts.size(),"model->outputs().size() != m_output_layouts.size()");
        for (int output_idx = 0; output_idx < m_model->get_output_size(); output_idx++)
        {
            auto output_layout = m_output_layouts[output_idx];
            
            ov::PartialShape output_partial_shape = m_model->output(output_idx).get_partial_shape();
            if (output_partial_shape.size() != output_layout.second.size()) {
                return_data.error_message = "<(E`_`E)> Please check your output_layout dims";
                return;
            }
            std::cout << "<(*^_^*)> Model output Shape: ";
            for (int i = 0; i < output_partial_shape.size(); i++)
            {
                std::cout << output_partial_shape[i] << (i != output_partial_shape.size() - 1 ? " x " : "");
            }
            std::cout << "   VS    Your output Shape: ";
            std::vector<ov::Dimension> output_dims;
            for (int i = 0; i < output_layout.second.size(); i++)
            {
                std::cout << output_layout.second[i] << (i != output_layout.second.size() - 1 ? " x " : "");
                auto dim = output_layout.second[i];
                if (dim == 0 || dim == -1) {
                    output_dims.push_back(ov::Dimension::dynamic());
                    continue;
                }
                output_dims.push_back(dim);
            }
            std::cout << "   LayOut:" << output_layout.first << std::endl;;
            ov::PartialShape output_shape = ov::PartialShape(output_dims);
            ppp.output(output_idx).tensor().set_element_type(ov::element::Type_t::f32).set_layout(ov::Layout(output_layout.first));
            std::cout << "<(*^_^*)> Output tensor has been setted on element_type/layout."<< std::endl;;
        }
        m_model = ppp.build();

        m_compiled_model = std::make_unique<ov::CompiledModel>(m_core->compile_model(m_model, m_device));
        std::cout << "<(*^_^*)> Compile_model tensor has been created successfully."<< std::endl;

        m_infer_request = std::make_unique<ov::InferRequest>(m_compiled_model->create_infer_request());
    },return_data);
    if (return_data.error_message.empty())
        return_data.result_state = true;
    return return_data;
}


bool OpenVinoInfer::Infer(const std::vector<std::vector<size_t>> &data_layouts,const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas){
    
    ResultData<bool> return_data;
    if (!m_infer_request) {
        std::cout <<  "<(E`_`E)> Please execute CreateEngine() first" << std::endl;
        return false;
    }
    InferenceCommon::TryFunction([&](){
        for (int input_idx = 0; input_idx < data_layouts.size(); input_idx++)
        {
            ov::Tensor input_tensor(ov::element::f32, ov::Shape(data_layouts[input_idx]), datas[input_idx]);
            m_infer_request->set_input_tensor(input_idx,input_tensor);
        }
        m_infer_request->infer();
        for (int output_idx = 0; output_idx < m_output_layouts.size(); output_idx++)
        {
            auto output = m_infer_request->get_output_tensor(output_idx);
            const float* output_buffer = output.data<const float>();
            auto out_shape = output.get_shape();
            std::cout << "(O_O)>>>>>  "  << output_idx+1 << "/" << m_output_layouts.size() << "   ";
            for (int i = 0; i < out_shape.size(); i++)
            {
                std::cout << out_shape[i] << (i!= out_shape.size()-1 ? " x " : "");
            }
            std::cout << std::endl;
            size_t element_count = output.get_size();
            std::vector<float> output_copy(element_count);
            std::copy_n(output_buffer, element_count, output_copy.data());
            output_datas.push_back(std::move(output_copy));
        }
    },return_data);
    if (return_data.error_message.empty()) {
        return_data.result_state = true;
    }
    return return_data.result_state;
}

ResultData<std::list<std::string>> OpenVinoInfer::GetInputNames(){
    ResultData<std::list<std::string>> return_data;
    return_data.result_info = std::list<std::string>();
    if (m_model) {
        for (int i = 0; i < m_model->inputs().size(); i++)
        {
            return_data.result_info.push_back(m_model->inputs()[i].get_any_name());
        }
    }

}

ResultData<std::list<std::string>> OpenVinoInfer::GetOutputNames(){
        ResultData<std::list<std::string>> return_data;
    return_data.result_info = std::list<std::string>();
    if (m_model) {
        for (int i = 0; i < m_model->outputs().size(); i++)
        {
            return_data.result_info.push_back(m_model->outputs()[i].get_any_name());
        }
    }
}

void OpenVinoInfer::ReleaseInferenceEngine(){
        std::cout << "<(*-_-*)> Releasing OpenVino inference engine..." << std::endl;
        if (m_infer_request) {
            m_infer_request = nullptr;
        }
        if (m_compiled_model) {
            m_compiled_model = nullptr;
        }
        if (m_model) {
            m_model = nullptr;
        }
}