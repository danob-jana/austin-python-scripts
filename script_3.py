#!/usr/bin/env python

from numbers import Number
import sklearn.datasets


def add_species_column(column_names, rows,
                       species_rows, species_names):
    column_names.append('species')
    for row, species_index in zip(rows, species_rows):
        row.append(species_names[species_index])


def compute_totals(column_names, rows):
    # compute the totals
    totals = [0 for column in column_names]
    for row in rows:
        for column_index, column_value in enumerate(row):
            # only sum number columns
            if isinstance(column_value, Number):
                totals[column_index] += column_value
    return totals


def load_and_extend_iris_data():
    '''
    returns a tuple, (column_names, rows, totals)
    '''
    raw_iris_data = sklearn.datasets.load_iris()
    rows = [list(r) for r in raw_iris_data.data]
    column_names = list(raw_iris_data['feature_names'])

    add_species_column(column_names, rows,
                       raw_iris_data.target,
                       raw_iris_data['target_names'])

    totals = compute_totals(column_names, rows)

    return (column_names, rows, totals)


def print_tab_separated(row):
    # note: doesn't properly handle cases where tabs are in the strings
    print('\t'.join([str(x) for x in row]))


def main():
    column_names, rows, totals = load_and_extend_iris_data()
    print_tab_separated(column_names)
    for row in rows:
        print_tab_separated(row)
    print_tab_separated(totals)


if __name__ == '__main__':
    main()
