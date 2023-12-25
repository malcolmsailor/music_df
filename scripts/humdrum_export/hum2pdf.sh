#!/usr/bin/env bash

# Get directory of this shell script:
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

set -e # abort on error

# IMG_CONVERTER=convert
IMG_CONVERTER=svgexport

if [[ -z $(which img2pdf) ]] || [[ -z $(which verovio) ]] || [[ -z $(which "$IMG_CONVERTER") ]]; then
    echo "ERROR: Missing a prerequisite: make sure img2pdf, verovio, and "
    echo "  ${IMG_CONVERTER} are in your path"
    exit 1
fi

input_krn="${1}"
output_pdf="${2}"
keep_intermediate_files="${3}"

if [[ -z "${input_krn}" ]]; then
    echo "Usage: hum2pdf <input_krn> <output_pdf>"
    echo "  If output_pdf is omitted, written to input_krn's path with "
    echo "  extension replaced with 'pdf'"
    exit 1
fi

if [[ -z "${output_pdf}" ]]; then
    output_pdf="${input_krn%.*}".pdf
fi

if [[ -z "${keep_intermediate_files}" ]]; then

    temp_dir=$(mktemp -d)

    # Make a temporary file with suffix "png" and store in color_legend variable
    color_dir=$(mktemp -d)

    # see comment at https://unix.stackexchange.com/a/181939/455517:
    #   You can use trap "rm -f $temp_file" 0 2 3 15 right after creating the file so that
    #   when the script exits or is stopped with ctrl-C the file is still removed.
    trap "rm -R $temp_dir" 0 2 3 15
    trap "rm -R $color_dir" 0 2 3 15
else
    temp_dir=~/tmp/hum2pdf
    color_dir=~/tmp/hum2pdf
    mkdir -p "${temp_dir}"
fi
color_legend="${color_dir}/color_legend.png"

set -x
python3 "${DIR}"/make_legend.py "${input_krn}" "${color_legend}"
verovio "${input_krn}" "-o" "${temp_dir}/tmp.svg" --footer none --header none --all-pages

for f in $(ls "${temp_dir}"/*.svg); do
    "${IMG_CONVERTER}" "$f" "${f/%svg/png}"
done

img2pdf $(ls "${color_dir}"/*.png) $(ls "${temp_dir}"/*.png) -o "${output_pdf}" >/dev/null 2>&1
set +x
