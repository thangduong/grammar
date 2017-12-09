#include "TensorflowModel.h"

TensorflowModel::TensorflowModel()
{
}


TensorflowModel::~TensorflowModel()
{
}

bool TensorflowModel::LoadGraphDef(const string& graphdef_filepath) {
	GraphDef graph_def;
	SessionOptions options;
	bool result;
	DWORD before_read_time = timeGetTime();
	
	Status status = ReadBinaryProto(Env::Default(), graphdef_filepath, &graph_def);
	if (status.ok()) {
		Session* session = 0;
		status = NewSession(options, &session);
		if (status.ok()) {
			_tf_session = shared_ptr<Session>(session);
			status = _tf_session->Create(graph_def);
			DWORD load_time = timeGetTime() - before_read_time;
			if (status.ok()) {
				char msg[128];
				sprintf(msg, "Load time: %d", load_time);
				DebugOut(msg);
				result = true;
			} else
				SetErrorMessage("Failed to load graph_def");
		}
		else {
			SetErrorMessage("Failed to create session");
		}
	}
	else {
		SetErrorMessage(string("Failed to load graphdef file: ") + graphdef_filepath);
	}
	return result;
}

void TensorflowModel::SetErrorMessage(const string& msg) {
	DebugOut(string("ERROR: ")+msg);
}
