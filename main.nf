#!/usr/bin/env nextflow

/// To use DSL-2 will need to include this
nextflow.enable.dsl=2

// Importing apply process
include { APPLY } from './modules/apply'

/// Print a header
log.info """\

=======================================================================================
MIBI apply model pipeline - nf 
=======================================================================================

Created by the Claire Marceaux, WEHI

Find documentation and more info @ GITHUB REPO DOT COM

Cite this pipeline @ INSERT DOI

Log issues @ GITHUB REPO DOT COM

=======================================================================================
Workflow run parameters 
=======================================================================================
input            : ${params.input}
run_name         : ${params.run_name}
model            : ${params.model}
preprocess_scheme: ${params.preprocess_scheme}
options_file     : ${params.options_file}
decoder_file     : ${params.decoder_file}
images_list      : ${params.images_list}
output_path      : ${params.output_path}
threshold        : ${params.threshold}
workDir          : ${workflow.workDir}
=======================================================================================

"""

/// Help function
def helpMessage() {
    log.info"""
  Usage:  nextflow run main.nf 
        --input <preprocessed-data.csv>
		--run_name <name>
		--model <bayes-cv-model.sav>
		--preprocess_scheme <scheme>
		--options_toml <preprocess-options.toml>
		--decoder_file <decoder.json>
		--images_list <images.csv>
		--output_path <output>
		--threshold <0.xxx>

  Required Arguments:

  --input			   Preprocessed input data file from QuPath.
  --run_name 		   Run name used to label output files.
  --model              Final model saved from training.
  --preprocess_scheme  The scheme used to preprocess the model before application.
  --options_toml       TOML file containing preprocessing scheme and model classifier options.
  --decoder_file       JSON file containing the decoder for the predicted cell types.
  --output_path        Path to directory to store output files.

  Optional Arguments:

  --threshold	       Not sure what this does yet
	
""".stripIndent()
}


workflow {
	// Show help message if --help is run or if any required params are not 
	// provided at runtime

	if ( params.help || 
	     params.input == "" ||
		 params.run_name == "" ||
         params.model == "" ||
         params.preprocess_scheme == "" ||
         params.options_file == "" ||
         params.decoder_file == "" ||
         params.images_list == "" ||
         params.output_path == ""){   
		
		// Invoke the help function above and exit
		helpMessage()
		exit 1

	// if none of the above are a problem, then run the workflow
	} else {
		
		// Define input channels 
		input_ch = Channel.fromPath("${params.input}")
		model_ch = Channel.fromPath("${params.model}")
		options_ch = Channel.fromPath("${params.options_file}")
		decoder_ch = Channel.fromPath("${params.decoder_file}")
		images_ch = Channel.fromPath("${params.images_list}")

		// Run process 1 example
		output_ch = APPLY(input_ch, model_ch, options_ch, decoder_ch, images_ch)
	}
}

workflow.onComplete {
summary = """
=======================================================================================
Workflow execution summary
=======================================================================================

Duration    : ${workflow.duration}
Success     : ${workflow.success}
workDir     : ${workflow.workDir}
Exit status : ${workflow.exitStatus}
outDir      : ${params.outDir}

=======================================================================================
  """
println summary

}
