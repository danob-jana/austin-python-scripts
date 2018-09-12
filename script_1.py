#!/usr/bin/env python

import sklearn.datasets


def main():
    raw_iris_data = sklearn.datasets.load_iris()
    rows = [list(r) for r in raw_iris_data.data]
    column_names = list(raw_iris_data['feature_names'])

    # add another column for 'species'. this shows up in another field called
    # 'target', but needs to be translated from a number into a name (the
    # number is an index into the target_names array)
    column_names.append('species')
    for row, species_index in zip(rows, raw_iris_data.target):
        row.append(raw_iris_data['target_names'][species_index])

    print('\t'.join(column_names))
    for row in rows:
        print('\t'.join([str(x) for x in row]))


if __name__ == '__main__':
    main()
