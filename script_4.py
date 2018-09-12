#!/usr/bin/env python

from numbers import Number
import sklearn.datasets


def format_row(row):
    # note: doesn't properly handle cases where tabs are in the strings, or
    # where unicode characters are involved.
    return '\t'.join([str(x) for x in row])


class IrisData:
    def __init__(self):
        self.raw_data = sklearn.datasets.load_iris()
        self.rows = [list(r) for r in self.raw_data.data]
        self.column_names = list(self.raw_data['feature_names'])

        self.add_species_column()
        self.compute_totals()

    def add_species_column(self):
        species_names = self.raw_data['target_names']
        self.column_names.append('species')
        for row, species_index in zip(self.rows, self.raw_data.target):
            row.append(species_names[species_index])

    def compute_totals(self):
        self.totals = [0 for column in self.column_names]
        for row in self.rows:
            for column_index, column_value in enumerate(row):
                # only sum number columns
                if isinstance(column_value, Number):
                    self.totals[column_index] += column_value

    def print_tab_separated(self):
        print(format_row(self.column_names))
        for row in self.rows:
            print(format_row(row))
        print(format_row(self.totals))


if __name__ == '__main__':
    data = IrisData()
    data.print_tab_separated()
