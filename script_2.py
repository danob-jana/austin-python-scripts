#!/usr/bin/env python

# you'll need to pip install scikit-learn
import sklearn.datasets


if __name__ == '__main__':
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

    # compute the totals
    totals = [0 for column in column_names]
    for row in rows:
        for column_index, column_value in enumerate(row):
            # only sum number columns
            if not isinstance(column_value, basestring):
                totals[column_index] += column_value

    # output the totals row
    print('\t'.join([str(x) for x in totals]))
