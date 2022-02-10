import json
import pickle
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

relations_prune = ['affects', 'augments', 'causes', 'interacts_with', 'part_of', 'precedes', 'predisposes', 'produces']

hf_icd = ['398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11',
          '404.13', '404.91', '404.93', '428', '428.0', '428.1', '428.2', '428.20',
          '428.21', '428.22', '428.23', '428.3', '428.30', '428.31', '428.32',
          '428.33', '428.4', '428.40', '428.41', '428.42', '428.43', '428.9']

copd_icd = ['490', '491', '491.0', '491.1', '491.2', '491.20', '491.21', '491.22',
        '491.8', '491.9', '492', '492.0', '492.8', '494', '494.0', '494.1', '496']

kidney_icd = ['285.21', '585', '585.1', '585.2', '585.3', '585.4', '585.5', '585.6', '585.9', '593.9'
         '753.10', '403', '403.0', '403.00', '403.01', '403.1', '403.10', '403.11', '403.9', '403.90', '403.91']

def map_icd2cui(snomedct_path: str, icd2snomed_1to1_path: str, icd2snomed_1toM_path: str):
    # TODO: optimizing the selection problem of 1toM mapping
    """
    convert icd-9 codes to umls_cui via snomed_cid

    `return`:
        icd2cui: dict: {"icd_code (str)": umls_cui (str)}
        icd_list: list of all icd codes in the icd2cui dictionary
    """
    # load snomed and icd2snomed
    snomed = pd.read_table(snomedct_path, sep="|")
    snomed2cui = snomed[["SNOMED_CID", "UMLS_CUI"]]
    icd2snomed_1to1 = pd.read_table(icd2snomed_1to1_path)
    icd2snomed_1toM = pd.read_table(icd2snomed_1toM_path)
    # convert the dataframe into dictionary
    snomed2cui_dict = snomed2cui.set_index("SNOMED_CID").T.to_dict("records")[0]  # dict: {"snomed_cid (int)": umls_cui (str)}
    # map cui to icd2snomed via snomed
    icd2snomed_1to1["UMLS_CUI"] = icd2snomed_1to1["SNOMED_CID"].map(snomed2cui_dict)
    icd2snomed_1toM["UMLS_CUI"] = icd2snomed_1toM["SNOMED_CID"].map(snomed2cui_dict)
    # drop all rows that have any NaN values
    icd2snomed_1to1 = icd2snomed_1to1.dropna(axis=0, how="any")
    icd2snomed_1toM = icd2snomed_1toM.dropna(axis=0, how="any")
    # extract icd and cui
    icd_cui_1to1 = icd2snomed_1to1[["ICD_CODE", "UMLS_CUI"]]
    icd_cui_1toM = icd2snomed_1toM[["ICD_CODE", "UMLS_CUI"]]
    # drop duplicates in icd codes
    icd_cui_1toM = icd_cui_1toM.drop_duplicates(subset=["ICD_CODE"], keep="first")
    # convert the dataframe into dictionary
    icd2cui_1to1 = icd_cui_1to1.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui_1toM = icd_cui_1toM.set_index("ICD_CODE").T.to_dict("records")[0]
    icd2cui = {}
    icd2cui.update(icd2cui_1to1)
    icd2cui.update(icd2cui_1toM)
    # make the list of all icd codes in the dictionary
    icd_list = list(icd2cui.keys())
    cui_list = list(icd2cui.values())
    return icd2cui, icd_list, cui_list


def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8: # there exists some strange cui with over 7 digs
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list


def extract_semmed_cui(semmed_csv_path, semmed_cui_path):
    """
    read the original SemMed csv file to extract all cui and store
    """
    print('extracting cui list from SemMed...')

    semmed_cui_list = []
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))

    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')

            if ls[4].startswith("C"):
                subj = ls[4]
                if len(subj) != 8:
                    subj_list = separate_semmed_cui(subj)
                    semmed_cui_list.extend(subj_list)
                else:
                    semmed_cui_list.append(subj)
            if ls[8].startswith("C"):
                obj = ls[8]
                if len(obj) != 8:
                    obj_list = separate_semmed_cui(obj)
                    semmed_cui_list.extend(obj_list)
                else:
                    semmed_cui_list.append(obj)

        semmed_cui_list = list(set(semmed_cui_list))

    with open(semmed_cui_path, "w", encoding="utf-8") as fout:
        for semmed_cui in semmed_cui_list:
            assert len(semmed_cui) == 8
            assert semmed_cui.startswith("C")
            fout.write(semmed_cui + "\n")

    print(f'extracted cui saved to {semmed_cui_path}')
    print()


def construct_graph(semmed_csv_path, semmed_cui_path, output_path):
    print("generating SemMed graph file...")

    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations_prune
    relation2idx = {r: i for i, r in enumerate(idx2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))

    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        attrs = set()
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls[3].lower() not in idx2relation:
                continue
            if ls[4] == ls[8]: # delete self-loop, not useful for our task
                continue
            sent = ls[1]
            rel = relation2idx[ls[3].lower()]

            if ls[4].startswith("C") and ls[8].startswith("C"):
                if len(ls[4]) == 8 and len(ls[8]) == 8:
                    subj = cui2idx[ls[4]]
                    obj = cui2idx[ls[8]]
                    if (subj, obj, rel) not in attrs:
                        graph.add_edge(subj, obj, rel=rel, sent=sent)
                        attrs.add((subj, obj, rel))
                elif len(ls[4]) != 8 and len(ls[8]) == 8:
                    cui_list = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list]
                    obj = cui2idx[ls[8]]
                    for subj in subj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                elif len(ls[4]) == 8 and len(ls[8]) != 8:
                    cui_list = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list]
                    subj = cui2idx[ls[4]]
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                else:
                    cui_list1 = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list1]
                    cui_list2 = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list2]
                    for subj in subj_list:
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                graph.add_edge(subj, obj, rel=rel, sent=sent)
                                attrs.add((subj, obj, rel))

    nx.write_gpickle(graph, output_path)

    # with open(output_txt_path, "w", encoding="utf-8") as fout:
    #     for triple in attrs:
    #         fout.write(str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])+ "\n")

    print(f"graph file saved to {output_path}")
    #print(f"txt file saved to {output_txt_path}")
    print()


def load_semmed(semmed_graph_path):
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, data in semmed.edges(data=True):
        w = 1.0 # initial weight to 1
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)
    return semmed_simple, semmed

def load_resources(semmed_cui_path):

    with open(semmed_cui_path, "r", encoding="utf8") as fin:
        id2cui = [c.strip().split("	")[0] for c in fin]
        #print(id2cui[0:10])
    cui2id = {c: i for i, c in enumerate(id2cui)}

    id2relation = relations_prune
    relation2id = {r: i for i, r in enumerate(id2relation)}
    return cui2id, id2cui, relation2id, id2relation

def relation_encoding(code2id, icd2cui, graph_path, semmed_cui_path, target_disease):
    semmed_simple, semmed = load_semmed(graph_path)
    cui2id, id2cui, relation2id, id2relation = load_resources(semmed_cui_path)
    rel_encoding1 = np.zeros((len(code2id)+1, 10, 1))
    rel_encoding2 = np.zeros((len(code2id)+1, 20, 2))

    if target_disease == 'Heart_failure':
        t_icds = hf_icd
    elif target_disease == 'COPD':
        t_icds = copd_icd
    else:
        t_icds = kidney_icd
    t_ids = []
    for icd in t_icds:
        if icd in icd2cui.keys():
            cui = icd2cui[icd]
            if cui in cui2id.keys():
                t_ids.append(cui2id[cui])

    # 1 hop paths
    i = 0
    for icd in code2id.keys():
        paths1 = []
        if icd in icd2cui.keys():
            cui = icd2cui[icd]
            if cui in cui2id.keys():
                cid = cui2id[cui]
                for tid in t_ids:
                    if semmed.has_edge(cid, tid):
                        for e_attr in semmed[cid][tid].values():
                            if e_attr['rel'] >= 0 and e_attr['rel'] < len(relation2id):
                                paths1.append(e_attr['rel'])
        for j in range(min(len(paths1), 10)):
            rel_encoding1[i, j] = paths1[j] + 1
        i = i + 1

    # 2 hop paths
    k = 0
    for icd in code2id.keys():
        paths2 = []
        if icd in icd2cui.keys():
            cui = icd2cui[icd]
            if cui in cui2id.keys():
                cid = cui2id[cui]
                for tid in t_ids:
                    extra_nodes = set()
                    if cid != tid and cid in semmed_simple.nodes and tid in semmed_simple.nodes:
                        extra_nodes |= set(semmed_simple[cid]) & set(semmed_simple[tid])
                        for node in extra_nodes:
                            if semmed.has_edge(cid, node) and semmed.has_edge(node, tid):
                                for e_attr in semmed[cid][node].values():
                                    for e_attr2 in semmed[node][tid].values():
                                        paths2.append([e_attr['rel'], e_attr2['rel']])
        for j in range(min(len(paths2), 20)):
            rel_encoding2[k, j, 0] = paths2[j][0] + 1
            rel_encoding2[k, j, 1] = paths2[j][1] + 1
        k = k + 1

    return rel_encoding1, rel_encoding2


def relation_encoding2(code2id, icd2cui, graph_path, semmed_cui_path, target_disease):
    semmed_simple, semmed = load_semmed(graph_path)
    cui2id, id2cui, relation2id, id2relation = load_resources(semmed_cui_path)
    rel_encoding = np.zeros((len(code2id)+1, 10, 1))

    if target_disease == 'Heart_failure':
        t_icds = hf_icd
    elif target_disease == 'COPD':
        t_icds = copd_icd
    else:
        t_icds = kidney_icd
    t_ids = []
    for icd in t_icds:
        if icd in icd2cui.keys():
            cui = icd2cui[icd]
            if cui in cui2id.keys():
                t_ids.append(cui2id[cui])

    # 1 hop paths
    i = 0
    for icd in code2id.keys():
        paths1 = []
        paths2 = []
        if icd in icd2cui.keys():
            cui = icd2cui[icd]
            if cui in cui2id.keys():
                cid = cui2id[cui]
                for tid in t_ids:

                    extra_nodes = set()
                    if cid != tid and cid in semmed_simple.nodes and tid in semmed_simple.nodes:
                        extra_nodes |= set(semmed_simple[cid]) & set(semmed_simple[tid])
                        for node in extra_nodes:
                            if semmed.has_edge(cid, node) and semmed.has_edge(node, tid):
                                for e_attr in semmed[cid][node].values():
                                    for e_attr2 in semmed[node][tid].values():
                                        paths2.append(2)

                    if semmed.has_edge(cid, tid):
                        for e_attr in semmed[cid][tid].values():
                            if e_attr['rel'] >= 0 and e_attr['rel'] < len(relation2id):
                                paths1.append(1)
        paths1.extend(paths2)
        for j in range(min(len(paths1), 10)):
            rel_encoding[i, j] = paths1[j]

        i = i + 1

    # 2 hop paths

    return rel_encoding


if __name__ == "__main__":
    icd2cui, icd_list, cui_list = map_icd2cui('../data/semmed/SNOMEDCT_CORE_SUBSET_202002.txt', '../data/semmed/ICD9CM_SNOMED_MAP_1TO1_201912.txt',
                                              '../data/semmed/ICD9CM_SNOMED_MAP_1TOM_201912.txt')
    pickle.dump(icd2cui, open('../data/semmed/icd2cui.pickle', 'wb'))
    # print(icd2cui)
    # print(icd_list)
    # print(cui_list)
    # extract_semmed_cui("../data/semmed/database.csv", "../data/semmed/cui_vocab.txt")
    # construct_graph("../data/semmed/database.csv", "../data/semmed/cui_vocab.txt", "../data/semmed/database_all.graph")



