Types
---------------

Base type to use with autofeasts. 


This will init a dataclass with 
all neccessary information to build features with autofeats.
This class also creates a join between table and public to
create a time window used in features creation.

The join works creating a on clause with the id (for example a customer_id)
and dates columns.

- Example:

   If you have a data_ref equals to **"2022-01-06"** and defines subtract_in_end
   parameter equals to 5 you will select all transactions from **"2022-01-01"** until
   **"2022-01-05"**.


.. automodule:: autofeats.types
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: dataclass