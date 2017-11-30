model_name = 'pptrankerV0'
params = {
	'model_name': model_name,
	'num_classes': 2,
	'feature_count': 3635,
	'mlp_config': [1024,1024,1024,1],
	'mlp_activations': 'sigmoid',
	'mlp_dropout_keep_probs': 0.5,
	'learning_rate': 0.001,
	'data_dir': './data/',
	'output_location': './output/'+model_name
}