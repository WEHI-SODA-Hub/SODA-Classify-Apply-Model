#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process APPLY {	
	cpus 1
	memory "4 GB"
	conda "${projectDir}/envs/environment.yml"
	publishDir "${params.output_path}", mode: 'copy'

	input:
	path(input_ch)
	path(model_ch)
	path(options_ch)
	path(decoder_ch)
	path(images_ch)

	output:
	path("*_applied_results.csv", type: "file")
	
	script:
	def options = options_ch.name != "NO_FILE" ? "--options ${options_ch}" : ''
	"""
	python3 ${projectDir}/scripts/apply_final_classifier.py \\
		--input ${input_ch} \\
		--name ${params.run_name} \\
		--model ${model_ch} \\
		--preprocess-scheme ${params.preprocess_scheme} \\
		--decoder ${decoder_ch} \\
		--images-file ${images_ch} \\
		--output-path . \\
		--threshold ${params.threshold} \\
		${options}
	"""
}