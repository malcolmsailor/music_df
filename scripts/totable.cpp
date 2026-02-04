#include <regex>
#include <string>

#include "humlib.h"

using namespace std;
using namespace hum;

vector<string> InstrumentCode;

void printHeader(bool has_label_ids) {
    if (has_label_ids) {
        cout << "track\tinstrument\tpitch\tonset\trelease\tspelling\ttype\tothe"
                "r\tlabel"
             << endl;
    } else {
        cout
            << "track\tinstrument\tpitch\tonset\trelease\tspelling\ttype\tother"
            << endl;
    }
}

void printRow(string track, string instrument, int pitch, float onset,
              float release, string spelling, string type_) {
    cout << track << '\t' << instrument << '\t' << pitch << '\t' << onset
         << '\t' << release << '\t' << spelling << '\t' << type_ << '\t' << ""
         << endl;
}

void printRow(string track, string instrument, int pitch, float onset,
              float release, string spelling, string type_, string labels) {
    cout << track << '\t' << instrument << '\t' << pitch << '\t' << onset
         << '\t' << release << '\t' << spelling << '\t' << type_ << '\t' << ""
         << '\t' << labels << endl;
}

void printRow(float onset, string type_, bool include_label_col) {
    if (include_label_col) {
        cout << "" << '\t' << "" << '\t' << "" << '\t' << onset << '\t' << ""
             << '\t' << "" << '\t' << type_ << '\t' << "" << '\t' << "" << endl;
    } else {
        cout << "" << '\t' << "" << '\t' << "" << '\t' << onset << '\t' << ""
             << '\t' << "" << '\t' << type_ << '\t' << "" << endl;
    }
}

void printRow(float onset, string type_, string other, bool include_label_col) {
    if (include_label_col) {
        cout << "" << '\t' << "" << '\t' << "" << '\t' << onset << '\t' << ""
             << '\t' << "" << '\t' << type_ << '\t' << other << '\t' << ""
             << endl;
    } else {
        cout << "" << '\t' << "" << '\t' << "" << '\t' << onset << '\t' << ""
             << '\t' << "" << '\t' << type_ << '\t' << other << endl;
    }
}

// TODO rename this function? Isn't it more like "printNote?"

void checkPitch(HTp token, int tpq, set<char> label_ids) {
    float onset = token->getDurationFromStart(tpq).getFloat() / tpq;
    float release = onset + token->getTiedDuration(tpq).getFloat() / tpq;
    int track_i = token->getTrack();
    vector<string> chordnotes = token->getSubtokens();
    for (size_t i = 0; i < chordnotes.size(); i++) {
        int pitch = Convert::kernToMidiNoteNumber(chordnotes[i]);
        string sciPitch = Convert::kernToSciPitch(chordnotes[i], "b", "#", ":");
        string spelling = regex_replace(sciPitch, regex(":([0-9])+$"), "");
        string track = token->getTrackString();
        string instrument = InstrumentCode.at(track_i);
        if (!label_ids.empty()) {
            string labels;
            for (char c : chordnotes[i]) {
                if (label_ids.find(c) != label_ids.end()) {
                    labels += c;
                }
            }
            printRow(track, instrument, pitch, onset, release, spelling, "note",
                     labels);
        } else {
            printRow(track, instrument, pitch, onset, release, spelling,
                     "note");
        }
    }
}

void printBarline(HumdrumLine line, int tpq, bool include_label_col) {
    float onset = line.getDurationFromStart(tpq).getFloat() / tpq;
    printRow(onset, "bar", include_label_col);
}

void printTimeSig(HTp token, int tpq, bool include_label_col) {
    float onset = token->getDurationFromStart(tpq).getFloat() / tpq;
    // we skip *M
    size_t i = 2;
    for (; i < token->length(); i++) {
        if ((*token)[i] == '/') {
            break;
        }
    }
    string numer = token->substr(2, i - 2);
    string denom = token->substr(i + 1);
    printRow(onset, "time_signature",
             "{\"numerator\": " + numer + ", \"denominator\": " + denom + "}",
             include_label_col);
}

void checkForInstrument(HumdrumLine& line) {
    HumRegex hre;
    for (int i = 0; i < line.getFieldCount(); i++) {
        HTp token = line.token(i);
        if (hre.search(token, "^\\*I([a-z][a-zA-Z0-9_-]*)$")) {
            string code = hre.getMatch(1);
            int track = token->getTrack();
            InstrumentCode.at(track) = code;
        }
    }
}

int main(int argc, char** argv) {
    if ((argc != 2) && (argc != 3)) {
        cout << "Usage: totable [kern file] {optional label identifiers}"
             << endl;
        return 1;
    }
    HumdrumFile infile;
    if (!infile.read(argv[1])) {
        return 1;
    }
    int tpq = infile.tpq();

    set<char> label_ids;
    if (argc > 2) {
        string label_ids_str(argv[2]);
        for (char c : label_ids_str) {
            label_ids.insert(c);
        }
    }

    printHeader(!label_ids.empty());

    InstrumentCode.resize(infile.getMaxTrack() + 1);
    for (int i = 1; i < (int)InstrumentCode.size(); i++) {
        InstrumentCode[i] = "none";
    }

    for (int i = 0; i < infile.getLineCount(); i++) {
        if (infile[i].isBarline()) {
            printBarline(infile[i], tpq, !label_ids.empty());
        }
        if (infile[i].isInterp()) {
            checkForInstrument(infile[i]);
            HTp token = infile.token(i, 0);
            if (token->isTimeSignature()) {
                // cout << "TIME SIG" << endl;
                printTimeSig(token, tpq, !label_ids.empty());
            }
        }
        if (!infile[i].isData()) {
            continue;
        }
        for (int j = 0; j < infile[i].getFieldCount(); j++) {
            HTp token = infile.token(i, j);
            // cout << token->getSpineInfo() << endl;
            if (!token->isKern()) {
                continue;
            }
            if (token->isNull() || token->isRest() ||
                token->isSecondaryTiedNote()) {
                continue;
            }
            checkPitch(token, tpq, label_ids);
        }
    }
    return 0;
}
