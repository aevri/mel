extern "C" {
#include <Python.h>
#include "structmember.h"
} // extern "C"

#include <set>
#include <utility>
#include <vector>

const int MAX_MOLE_COST = 10000;

typedef int MoleIndex;
typedef std::vector<MoleIndex> MoleIndexVector;
typedef std::set<MoleIndex> MoleIndexSet;

struct Mole {
    Mole() {
        this->loc = -1;
        this->ident = -1;
    }
    Mole(MoleIndex loc, MoleIndex ident) {
        this->loc = loc;
        this->ident = ident;
    }

    MoleIndex loc;
    MoleIndex ident;
};

struct Guess {
    Guess(MoleIndex ident, int cost) {
        this->ident = ident;
        this->cost = cost;
    }

    MoleIndex ident;
    int cost;
};

typedef std::vector<Guess> GuessVector;

bool pyobject_to_guessvector(PyObject *list, GuessVector& result) {
    PyObject *fast_list = PySequence_Fast(
        list,
        "Must be a sequence.");

    if (! fast_list) {
        return false;
    }

    PyObject **fast_items = PySequence_Fast_ITEMS(fast_list);

    Py_ssize_t fast_size = PySequence_Fast_GET_SIZE(fast_list);

    result.clear();

    /* printf("Guesses ["); */
    for (int i=0; i < fast_size; ++i) {
        PyObject *item = fast_items[i];

        if (! PyTuple_Check(item)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Guess items must be tuples."
            );
            return false;
        }

        if (2 != PyTuple_Size(item)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Guess items must be tuples of size 2."
            );
            return false;
        }

        result.push_back(
            Guess(
                PyLong_AsLong(PyTuple_GetItem(item, 0)),
                PyLong_AsLong(PyTuple_GetItem(item, 1))
            )
        );

        /* printf("(%i, %i) ", result.back().ident, result.back().cost); */

        if (PyErr_Occurred()) {
            return false;
        }
    }

    /* printf("]\n"); */
    return true;
}

struct CalcGuessesFunctor {
    CalcGuessesFunctor(
        PyObject *callable,
        int num_canonical,
        int num_locations,
        int num_identities)
    {
        this->calc_guesses_fn = callable;
        Py_INCREF(this->calc_guesses_fn);

        // TODO: Take advantage of the fact that (predictor_loc, guess_loc)
        // pairs are always the same. There is only one predictor for each
        // guess_loc.

        // TODO: Don't include canonical indices in the cache, they will never
        // be queried. Even if they were, the cost would always be 1.

        const int cache_size = num_locations * num_identities * num_locations;
        this->guess_cache.resize(cache_size);
        this->guess_cache_has_guess.resize(cache_size);

        this->num_locations = num_locations;
        this->num_identities = num_identities;

        for (auto i : this->guess_cache_has_guess) {
            if (i == true) {
                printf("ERROR: guess_cache_has_guess non-zero to start.");
                return;
            }
        }
    }

    ~CalcGuessesFunctor() {
        Py_XDECREF(this->calc_guesses_fn);
    }

    const GuessVector&
    operator()(Mole predictor, MoleIndex guess_loc) const {
        const int cache_index = (
            predictor.loc +
            (predictor.ident * this->num_locations) +
            (guess_loc * this->num_locations * this->num_identities));
        GuessVector& guesses = this->guess_cache[cache_index];

        /* printf( */
        /*     "(%i, %i) %i ...\n", */
        /*     predictor.loc, predictor.ident, guess_loc); */

        if (this->guess_cache_has_guess[cache_index] == true) {
            /* printf( */
            /*     "Cached (%i, %i) %i\n", */
            /*     predictor.loc, predictor.ident, guess_loc); */
            return guesses;
        }

        if (PyErr_Occurred()) {
            printf("Error before trying to get guesses.\n");
            return guesses;
        }

        PyObject *args = Py_BuildValue(
            "((ii)i)", predictor.loc, predictor.ident, guess_loc);
        if (NULL == args) {
            printf("Error encountered creating args for calc_guesses.\n");
            return guesses;
        }

        PyObject *result = PyObject_CallObject(this->calc_guesses_fn, args);
        Py_DECREF(args);
        if (NULL == result) {
            printf("Error encountered calling calc_guesses.\n");
            return guesses;
        }

        /* printf("(%i, %i) %i -> ", predictor.loc, predictor.ident, guess_loc); */
        if (! pyobject_to_guessvector(result, guesses)) {
            printf("Error encountered converting guesses.\n");
        }

        this->guess_cache_has_guess[cache_index] = true;

        return guesses;
    }

private:
    PyObject *calc_guesses_fn;

    mutable std::vector<GuessVector> guess_cache;
    mutable std::vector<bool> guess_cache_has_guess;

    int num_locations;
    int num_identities;
};

bool pyobject_to_moleindexvector(PyObject *list, MoleIndexVector& result) {
    PyObject *fast_list = PySequence_Fast(
        list,
        "Must be a sequence.");

    if (! fast_list) {
        return false;
    }

    PyObject **fast_items = PySequence_Fast_ITEMS(fast_list);

    Py_ssize_t fast_size = PySequence_Fast_GET_SIZE(fast_list);

    result.resize(fast_size);

    for (int i=0; i < fast_size; ++i) {
        PyObject *item = fast_items[i];

        if (item == Py_None) {
            result[i] = -1;
            continue;
        }

        if (! PyLong_Check(item)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Items must be longs or None."
            );
            return false;
        }

        result[i] = PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            return false;
        }
    }

    return true;
}

struct BounderCpp {
    BounderCpp(
        MoleIndexVector&& predictors,
        const CalcGuessesFunctor& calc_guesses,
        const int num_identities,
        const int num_canonicals)
        : predictors(std::move(predictors))
        , calc_guesses(calc_guesses)
        , num_identities(num_identities)
        , num_canonicals(num_canonicals)
    {
    }

    std::vector<int>
    lower_bound(const MoleIndexVector& state) const {
        const int num_locations = state.size();
        MoleIndexVector available_idents;
        MoleIndexSet used_idents;

        /* printf("state: ["); */
        /* for (MoleIndex guess_loc=0; guess_loc < num_locations; ++guess_loc) { */
        /*     printf("%i ", state[guess_loc]); */
        /* } */
        /* printf("]\n"); */

        for (MoleIndex guess_loc=0; guess_loc < num_locations; ++guess_loc) {
            const MoleIndex guess_ident = state[guess_loc];
            if (guess_ident != -1) {
                used_idents.insert(guess_ident);
            }
        }

        for (
            MoleIndex ident=0;
            ident < this->num_identities;
            ++ident)
        {
            if (used_idents.find(ident) == used_idents.end()) {
                available_idents.push_back(ident);
            }
        }

        std::vector<int> lb(num_locations);

        /* printf("num_canonicals: %i\n", this->num_canonicals); */

        for (MoleIndex guess_loc=0; guess_loc < num_locations; ++guess_loc) {
            /* printf("location: %i\n", guess_loc); */
            if (guess_loc < this->num_canonicals) {
                lb[guess_loc] = 1;
                continue;
            }

            const MoleIndex guess_ident = state[guess_loc];
            const Mole guess(guess_loc, guess_ident);

            const MoleIndex predictor_loc = this->predictors[guess_loc];
            const MoleIndex predictor_ident = state[predictor_loc];
            const Mole predictor(predictor_loc, predictor_ident);

            if (guess_ident != -1) {
                if (predictor_ident != -1) {
                    lb[guess_loc] = this->cost_for_guess(predictor, guess);
                } else {
                    lb[guess_loc] = this->lower_bound_unk_predictor(
                        available_idents, predictor_loc, guess);
                }
            } else {
                if (predictor_ident != -1) {
                    lb[guess_loc] = this->lower_bound_unk_guess(
                        used_idents, predictor, guess_loc);
                } else {
                    lb[guess_loc] = this->lower_bound_unk_unk(
                        used_idents,
                        available_idents,
                        predictor_loc,
                        guess_loc);
                }
            }
        }

        return lb;
    }

private:

    int cost_for_guess(Mole predictor, Mole guess) const {
        /* printf("cost_for_guess (%i %i) (%i %i)\n", */
            /* predictor.loc, predictor.ident, guess.loc, guess.ident); */
        const auto guesses = this->calc_guesses(predictor, guess.loc);
        int cost = MAX_MOLE_COST;
        for (Guess g : guesses) {
            if (g.ident == guess.ident) {
                cost = g.cost;
                break;
            }
        }
        return cost;
    }

    int lower_bound_unk_predictor(
        const MoleIndexVector& possible_predictor_idents,
        MoleIndex predictor_loc,
        Mole guess) const
    {
        /* printf("lower_bound_unk_predictor %i (%i %i)\n", */
            /* predictor_loc, guess.loc, guess.ident); */
        int min_cost = MAX_MOLE_COST;
        for (MoleIndex predictor_ident : possible_predictor_idents) {
            const int cost = this->cost_for_guess(
                Mole(predictor_loc, predictor_ident), guess);
            if (cost < min_cost) {
                min_cost = cost;
            }
        }
        return min_cost;
    }

    int lower_bound_unk_guess(
        const MoleIndexSet& used_idents,
        Mole predictor,
        MoleIndex guess_loc) const
    {
        /* printf("lower_bound_unk_guess (%i %i) %i\n", */
            /* predictor.loc, predictor.ident, guess_loc); */
        const auto guesses = this->calc_guesses(predictor, guess_loc);
        int cost = MAX_MOLE_COST;
        for (Guess g : guesses) {
            if (used_idents.find(g.ident) == used_idents.end()) {
                cost = g.cost;
                break;
            }
        }
        return cost;
    }

    int lower_bound_unk_unk(
        const MoleIndexSet& used_idents,
        const MoleIndexVector& possible_predictor_idents,
        MoleIndex predictor_loc,
        MoleIndex guess_loc) const
    {
        /* printf("lower_bound_unk_unk %i %i\n", predictor_loc, guess_loc); */
        int min_cost = MAX_MOLE_COST;
        for (MoleIndex predictor_ident : possible_predictor_idents) {
            const int cost = this->lower_bound_unk_guess(
                used_idents, Mole(predictor_loc, predictor_ident), guess_loc);
            if (cost < min_cost) {
                min_cost = cost;
            }
        }
        return min_cost;
    }

    const MoleIndexVector predictors;
    const CalcGuessesFunctor& calc_guesses;
    const int num_identities;
    const int num_canonicals;
};

extern "C" {

typedef struct {
    PyObject_HEAD
    CalcGuessesFunctor *calc_guesses;
    BounderCpp *bounder;
} BounderPy;

static void
BounderPy_dealloc(BounderPy* self)
{
    if (self->bounder != NULL) {
        delete self->bounder;
        self->bounder = NULL;
    }
    if (self->calc_guesses != NULL) {
        delete self->calc_guesses;
        self->calc_guesses = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
BounderPy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BounderPy *const self = (BounderPy *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->calc_guesses = NULL;
        self->bounder = NULL;
    }

    return (PyObject *)self;
}

static int
BounderPy_init(BounderPy *self, PyObject *args, PyObject *kwds)
{
    PyObject *location_to_predictor = NULL;
    PyObject *calc_guesses = NULL;
    int num_identities;
    int num_canonicals;

    static char *kwlist[] = {
        "location_to_predictor",
        "calc_guesses",
        "num_identities",
        "num_canonicals",
        NULL
    };

    if (! PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOii:Bounder.__init__",
            kwlist,
            &location_to_predictor,
            &calc_guesses,
            &num_identities,
            &num_canonicals))
    {
        return -1;
    }

    MoleIndexVector predictors;
    if (!  pyobject_to_moleindexvector(location_to_predictor, predictors)) {
        return -1;
    }

    if (! PyCallable_Check(calc_guesses)) {
        PyErr_SetString(PyExc_TypeError, "'calc_guesses' must be callable.");
        return -1;
    }

    const int num_locations = predictors.size();
    self->calc_guesses = new CalcGuessesFunctor(
        calc_guesses, num_canonicals, num_locations, num_identities);
    self->bounder = new BounderCpp(
        std::move(predictors),
        *(self->calc_guesses),
        num_identities,
        num_canonicals);

    return 0;
}

static PyMemberDef BounderPy_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
BounderPy_lower_bound(BounderPy *self, PyObject *args)
{
    PyObject *state;
    if (! PyArg_ParseTuple(
            args,
            "O:Bounder.lower_bound",
            &state))
    {
        return NULL;
    }

    MoleIndexVector state_vector;
    if (! pyobject_to_moleindexvector(state, state_vector)) {
        return NULL;
    }

    std::vector<int> result = self->bounder->lower_bound(state_vector);

    PyObject *total = PyLong_FromLong(1);
    for (int i : result) {
        /* printf("%i ", i); */
        PyObject *item = PyLong_FromLong(i);
        PyObject *tmp = PyNumber_Multiply(total, item);
        Py_DECREF(item);
        Py_DECREF(total);
        if (NULL == tmp) {
            PyErr_SetString(
                PyExc_TypeError,
                "Failed to multiply results together.");
            return NULL;
        }
        total = tmp;
    }
    /* printf("\n"); */

    if (PyErr_Occurred()) {
        return NULL;
    }

    return total;
}

static PyMethodDef BounderPy_methods[] = {
    {
        "lower_bound",
        (PyCFunction)BounderPy_lower_bound,
        METH_VARARGS,
        "<< Lowerbound >>."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject BounderPyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "mel.rotomap.lowerbound.Bounder",  /* tp_name */
    sizeof(BounderPy),                 /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)BounderPy_dealloc,     /* tp_dealloc */
    0,                                 /* tp_print */
    0,                                 /* tp_getattr */
    0,                                 /* tp_setattr */
    0,                                 /* tp_reserved */
    0,                                 /* tp_repr */
    0,                                 /* tp_as_number */
    0,                                 /* tp_as_sequence */
    0,                                 /* tp_as_mapping */
    0,                                 /* tp_hash  */
    0,                                 /* tp_call */
    0,                                 /* tp_str */
    0,                                 /* tp_getattro */
    0,                                 /* tp_setattro */
    0,                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,           /* tp_flags */
    "Bounder class",                   /* tp_doc */
    0,                                 /* tp_traverse */
    0,                                 /* tp_clear */
    0,                                 /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    BounderPy_methods,                 /* tp_methods */
    BounderPy_members,                 /* tp_members */
    0,                                 /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)BounderPy_init,          /* tp_init */
    0,                                 /* tp_alloc */
    BounderPy_new,                     /* tp_new */
};

static PyModuleDef lowerbound_module = {
    PyModuleDef_HEAD_INIT,
    "lowerbound",
    "Implements mel.rotomap.identify.Bounder in C++.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

} // extern "C"

PyMODINIT_FUNC
PyInit_lowerbound(void)
{
    PyObject* m;

    if (PyType_Ready(&BounderPyType) < 0)
        return NULL;

    m = PyModule_Create(&lowerbound_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BounderPyType);
    PyModule_AddObject(m, "Bounder", (PyObject *)&BounderPyType);
    return m;
}
