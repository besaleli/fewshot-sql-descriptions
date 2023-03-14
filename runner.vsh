#!/usr/bin/env -S v

fn sh(cmd string) {
	println('RUNNING CMD $cmd')
	print(execute_or_exit(cmd).output)
}

collection_types := ['random', 'tfidf']
model_name := 'text-davinci-003'

for collection_type in collection_types {
	for nshot in 0 .. 4 {
		output_file := 'results/${model_name}-${collection_type}-${nshot}shot.json'

		cmd := [
			'python3 run_experiment.py',
			'--collection $collection_type',
			'--model $model_name',
			'--random_state 42',
			'--sample 50',
			'--nshot $nshot', 
			'-o $output_file',
			].join(' ')

		// don't do zero-shot multiple times
		if (nshot == 0 && collection_type == 'random') || (nshot > 0) {
			sh(cmd)
		}
	}
}
