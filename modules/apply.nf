#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process APPLY {	
	cpus 1
	memory "${params.memory}"
	conda "${projectDir}/envs/environment.yml"
	publishDir "${params.output_path}", mode: 'copy'
	label "apply"

	input:
	path(apply_script_ch)
	path("input.csv")
	path(model_ch)
	path(options_ch)
	path(decoder_ch)
	path(images_ch)
	path("validation.csv")
	val(threshold_ch)

	output:
	path("*_applied_results.csv", type: "file")
	
	script:
	def options = options_ch.name != "NO_FILE" ? "--options ${options_ch}" : ''
	def threshold = threshold_ch != "" ? "--threshold ${threshold_ch}" : ''
	"""
	python3 ${apply_script_ch} \\
		--input input.csv \\
		--name ${params.run_name} \\
		--model ${model_ch} \\
		--preprocess-scheme ${params.preprocess_scheme} \\
		--decoder ${decoder_ch} \\
		--images-file ${images_ch} \\
		--validation-file validation.csv \\
		--output-path . \\
		${threshold} ${options}
	"""
}
