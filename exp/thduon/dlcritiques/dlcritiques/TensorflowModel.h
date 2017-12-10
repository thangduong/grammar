#pragma once
#define COMPILER_MSVC
#define NOMINMAX
#include <Windows.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <memory>
#include <eigen/Dense>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
using namespace tensorflow;

class TensorflowModel
{
public:
	TensorflowModel();
	virtual ~TensorflowModel();

	bool LoadGraphDef(const string& graphdef_filepath);
protected:
	void SetErrorMessage(const string& msg);
	shared_ptr<Session> _tf_session;
	string _last_error_msg;
};

inline void DebugOut(const string& msg) {
	wstring wmsg;
	wmsg.assign(msg.begin(), msg.end());
	OutputDebugString(wmsg.c_str());
	OutputDebugString(L"\r\n");
}


