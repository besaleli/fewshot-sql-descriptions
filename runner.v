import os

fn sh(cmd string) {
	println('RUNNING CMD $cmd')
	print(os.execute_or_exit(cmd).output)
}

collection_types := ['random', 'column_jaccard', 'tfidf']
model_name := 'text-curie-001'

for collection_type in collection_types {
	for nshot in 0 .. 4 {
		for mask in [0, 1, 2, 1000] {
			output_file := 'results/${model_name}-${collection_type}-${nshot}shot-${mask}mask.json'

			cmd := [
				'python3 run_experiment.py',
				'--collection $collection_type',
				'--model $model_name',
				'--random_state 42',
				'--sample 50',
				'--mask_columns $mask',
				'--nshot $nshot', 
				'-o $output_file',
				].join(' ')

			// don't do zero-shot multiple times
			if (nshot == 0 && collection_type == 'random') || (nshot > 0) {
				sh(cmd)
			}
		}
	}
}
