import csv
import os
import json
from app.backend.helper import *


class preProcessing:
    def __init__(self, language):
        self.language = language

    def preProcessCONANDataSet(self):
        lexicon_filename = "new_hurtlex_{0}.tsv".format(self.language)
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                     'CONAN_dataSet.json')
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'improvedHurtlexLexica', self.language,
                                      lexicon_filename)

        if not os.path.exists(dest_file_path):
            pref_list = ['ENT', 'ITT']
            with open(dest_file_path, 'w') as dest_file:
                with open(src_file_path, 'r') as source_file:
                    element = json.load(source_file)
                    element = element['conan']
                    dest_file.write('hateSpeech\thsType\thsSubType\n')
                    for line in element:
                        if not line['cn_id'].startswith(tuple(pref_list)):
                            dest_file.write(str(line['hateSpeech'].replace("'", "’").replace('"', " ")) + '\t' + str(
                                line['hsType']) + '\t' + str(line['hsSubType']) + '\n')
            # remove duplicated lines
            uniqlines = set(open(dest_file_path).readlines())
            open(dest_file_path, 'w').writelines(set(uniqlines))

    def normalizeTweetsUsingNltk(self):
        lexicon_src_filename = 'train_data_for_pre_processing_' + self.language + '.tsv'
        lexicon_dest_filename = 'train_data_' + self.language + '.tsv'
        src_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language, lexicon_src_filename)
        dest_file_path = os.path.join(os.path.dirname(__file__), '../..', 'dataSet', self.language,
                                      lexicon_dest_filename)
        if not os.path.exists(dest_file_path):
            with open(dest_file_path, 'w') as dest_file:
                with open(src_file_path, 'r') as src_file:
                    if self.language == ENGLISH:
                        tsvreader = csv.reader(src_file, delimiter="\t")
                        id = 1
                        for line in tsvreader:
                            processed_tweet = normalizer(line[1], self.language)
                            if processed_tweet != '':
                                dest_file.write(str(id) + '\t' + processed_tweet + '\t' + str(line[2]) + '\n')
                                id += 1
                    else:
                        tsvreader = csv.reader(src_file, delimiter=",")
                        id = 1
                        for line in tsvreader:
                            processed_tweet = normalizer(line[1], self.language)
                            if processed_tweet != '':
                                if line[2] == 'normal':
                                    label = '0'
                                else:
                                    label = '1'
                                dest_file.write(str(id) + '\t' + processed_tweet + '\t' + label + '\n')
                                id += 1


if __name__ == "__main__":
    preProcessing = preProcessing(FRENCH)
    preProcessing.normalizeTweetsUsingNltk()
