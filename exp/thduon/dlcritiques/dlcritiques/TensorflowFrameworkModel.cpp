#include "TensorflowFrameworkModel.h"
#include <fstream>

TensorflowFrameworkModel::TensorflowFrameworkModel()
{
}


TensorflowFrameworkModel::~TensorflowFrameworkModel()
{
}

bool TensorflowFrameworkModel::Load(const string& params_json_filepath, const string& graphdef_filepath) {
	Json::Reader reader;
	ifstream jsonFile(params_json_filepath, ios_base::in);
	if (LoadGraphDef(graphdef_filepath)) {
		if (jsonFile.is_open()) {
			if (reader.parse(jsonFile, _params)) {
				DebugOut(string("Loaded: ") + params_json_filepath);
				return true;
			}
			else {
				SetErrorMessage(string("Failed to parse: ") + params_json_filepath);
				return false;
			}
		}
		else {
			SetErrorMessage(string("Failed to open: ") + params_json_filepath);
			return true;
		}
	}
	else {
		SetErrorMessage(string("Failed to load: ") + graphdef_filepath);
		return false;
	}
}
