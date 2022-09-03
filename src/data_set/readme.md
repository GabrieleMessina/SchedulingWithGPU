# DataSet: Scheduling with GPU

## DataSet organization
**#nodes** followed by #nodes rows with the following data:
|job_weight(mean of processors cost) | cost on processors |   #successors  | <successor_id, data to transfer> |  |
|------| ----------------|--------------------------------|------------|
|   1  |       3        |      <2, 1> <3, 5> <4, 3>      |            |
|   2  |       2        |        <12, 12> <75, 8>        |            |
|  ... |      ...       |               ...              |            |
|      |                |                                |            |
