
#pragma once
#include <openvino/op/op.hpp>
namespace TemplateExtension{

class AddGridSamplePlugin: public ov::op::Op
{

public:
    OPENVINO_OP("MyGridSample");
    AddGridSamplePlugin() = default;
    AddGridSamplePlugin(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& grid);
    void validate_and_infer_types();
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& inputs) const;
    bool visit_attributes(ov::AttributeVisitor& visitor);
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const; 
    bool has_evaluate() const;
};

} // namespace TemplateExtension