# DataSet: Scheduling with GPU

## DataSet organization
                                                                ?
|jobnr.|   #successors  | <successors, data to transfer> | complexity |
|------|----------------|--------------------------------|------------|
|   1  |       3        |      <2, 1> <3, 5> <4, 3>      |            |
|   2  |       2        |        <12, 12> <75, 8>        |            |
|  ... |      ...       |               ...              |            |
|  END |                |                                |            |
