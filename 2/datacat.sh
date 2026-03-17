#!/bin/bash
# simple script to combine multiple json dataset files into one
# takes list of lang labels as input and returns combined 

# valid arg options
VALID="de|en|es|ru|zh"

# exits if less than 2 args
(( $# < 2 )) && { echo "ERROR: Expects at least two langs in: $VALID"; exit 1; }

# data directory
dir="/fp/projects01/ec403/IN5550/obligatories/2"
# (prefix for) out directory
out_dir="data/"

# assoc array for keeping track of seen args
declare -A seen

# arrays for keeping track of files to zcat
devs=()
trains=()

for lang in "$@"; do
	case "$lang" in
		# lang arguments has to be in our set of options
		de|en|es|ru|zh)
			# if key exists (i.e. duplicate lang) we exit
			if [[ -v seen["$lang"] ]]; then
				echo "ERROR: No duplicates allowed, '$lang' occurs two times"
				exit 1
			fi
			# keeps track of seen lang args
			seen["$lang"]=1
			
			# output dir is formed by combining all langs with _ as sep
			out_dir+="$lang""_"

			devs+=("$dir"/"$lang""_dev.jsonl.gz")
			trains+=("$dir"/"$lang""_train.jsonl.gz")

			;;
		*)
			# if at least one given lang is not valid we
			echo "ERROR: '$lang' is not a valid lang."
			echo "Expects at least two langs in: $VALID"
			exit 1
			;;
	esac
done

# slice the dir with one to remove trailing underscore
out_dir="${out_dir::-1}"

# exits if the datasets have already been concatenated (i.e. if out_dir exists)
[[ -d "$out_dir" ]] && { echo "ERROR: Datacat '$out_dir' already exist. Have a nice day."; exit 1; }

# if not out dir is created we start by creating it
mkdir $out_dir

# filenames for output
out_train="$out_dir"/"train.jsonl"
out_dev="$out_dir"/"dev.jsonl"

# concatenates splits of given datasets to files
zcat ${devs[@]} > $out_dev
zcat ${trains[@]} > $out_train

echo "Datasets successfully concatenated. hjppi"
