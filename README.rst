=====
pgtaa
=====


.. image:: https://img.shields.io/pypi/v/pgtaa.svg
        :target: https://pypi.python.org/pypi/pgtaa

.. image:: https://img.shields.io/travis/SvenBecker/pgtaa.svg
        :target: https://travis-ci.org/SvenBecker/pgtaa

.. image:: https://readthedocs.org/projects/pgtaa/badge/?version=latest
        :target: https://pgtaa.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


A Policy Gradient Approach for Tactical Asset Allocation
------------------------------------------------------



Data
----

Data can be requested using `pandas_datareader <https://pandas-datareader.readthedocs.io/en/latest/>`_.
So far this project support `Yahoo Finance <https://finance.yahoo.com/>`_,
`Federal Reserve of Economic Data (FRED) <https://www.stlouisfed.org/>`_,
`The Investors Exchange <https://iextrading.com/>`_,
`Moscow Exchange <https://www.moex.com/en/>`_ and `Stooq <https://stooq.com/>`_
as data resources. Only the (adjusted if available) closing prices will be tracked.

..
Notice: Unfortunately there are often changes in the APIs of those website so I can't guarantee that those sites are
still being supported some time onwards. At the time writing those supported websites are indeed working as intended.



* Free software: Apache Software License 2.0
* Documentation: https://pgtaa.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
