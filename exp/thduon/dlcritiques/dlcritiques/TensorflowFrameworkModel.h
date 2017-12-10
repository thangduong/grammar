#pragma once
#include "TensorflowModel.h"
#include "json/json.h"

class TensorflowFrameworkModel :
	public TensorflowModel
{

public:
	TensorflowFrameworkModel();
	virtual ~TensorflowFrameworkModel();
	virtual bool Load(const string& params_json_filepath, const string& graphdef_filepath);
protected:
	Json::Value _params;
};

