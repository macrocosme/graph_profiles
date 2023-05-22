import numpy as np
from .positioning import get_weight

# Prediction section
def predict(population, k, neighbours, mst):
    def del_na(_classes:list):
        where_na = lambda _classes: (c == 'N/A' for c in _classes)
        for i, d in enumerate(where_na(_classes)):
            if d:
                del _classes[i]
        return _classes

    def majority(classes, verbose=False):
        # u: unique classes, c: unique counts
        u, c = np.unique(classes, return_counts=True)
        uu, cc = np.unique(c, return_counts=True)
        if verbose:
            msg = f"{uu}, {c}:"
            print (uu, c)
        if cc[np.where(uu == np.max(c))] == 1:
            maj = u[np.argmax(c)]
            if verbose:
                msg += f" {maj}"
        else:
            maj = None
            if verbose:
                msg += " 'No majority'"

        if verbose:
            print (msg)

        return maj

    classes = del_na([population.as_array()[c].morphological_class \
               if population.as_array()[c].morphological_class != 'N/A' \
               else population.as_array()[c].predicted_class \
               for c in neighbours[k]])

    if len(classes) > 0:
        if majority(classes) is not None:
            return majority(classes)
            # print (k, majority(classes))
        else:
            # find nearest neighbour
            nn = neighbours[k][np.argmin([get_weight(mst, k, c) for c in neighbours[k]])]
            return population.as_array()[nn].morphological_class
    else:
        nn = neighbours[k][np.argmin([get_weight(mst, k, c) for c in neighbours[k]])]
        return population.as_array()[nn].morphological_class
