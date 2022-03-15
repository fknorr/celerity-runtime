# Benchmark Results

| Metadata |                      |
| :------- | :------------------- |
| Created  | 2022-03-30T09:41:49Z |


| Test case                                                                                                                                         | Benchmark name                   |            Min |           Mean |       Std dev |
| :------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------- | -------------: | -------------: | ------------: |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | creating nodes                   |           4.47 |           4.72 |          0.18 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | creating and adding dependencies |          23.89 |          23.90 |          0.03 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | adding and removing dependencies |          20.52 |          20.58 |          0.41 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | checking for dependencies        |           2.09 |           2.10 |          0.05 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | creating nodes                   |          40.32 |          40.45 |          0.56 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | creating and adding dependencies |         279.61 |         280.83 |          0.68 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | adding and removing dependencies |         243.41 |         244.41 |          0.96 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | checking for dependencies        |          40.07 |          40.56 |          1.42 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | creating nodes                   |         442.34 |         444.49 |          7.66 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | creating and adding dependencies |       4'658.50 |       4'708.45 |        193.58 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | adding and removing dependencies |       4'737.00 |       4'784.50 |        157.63 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | checking for dependencies        |       1'999.85 |       2'019.68 |         76.35 |
| generating large task graphs                                                                                                                      | soup topology                    |   9'644'651.00 |  10'507'581.23 |  1'344'264.95 |
| generating large task graphs                                                                                                                      | chain topology                   |      69'850.00 |      72'003.62 |      3'597.93 |
| generating large task graphs                                                                                                                      | expanding tree topology          |     101'801.00 |     115'293.31 |     25'135.66 |
| generating large task graphs                                                                                                                      | contracting tree topology        |     180'569.00 |     182'230.92 |      2'539.45 |
| generating large task graphs                                                                                                                      | wave\_sim topology                |     623'837.00 |     638'478.62 |     12'501.54 |
| generating large task graphs                                                                                                                      | jacobi topology                  |     203'202.00 |     212'303.98 |     12'507.56 |
| generating large command graphs for N nodes - 1                                                                                                   | soup topology                    |  16'089'750.00 |  18'377'377.10 |    972'728.92 |
| generating large command graphs for N nodes - 1                                                                                                   | chain topology                   |     279'897.00 |     281'850.74 |      3'284.19 |
| generating large command graphs for N nodes - 1                                                                                                   | expanding tree topology          |     343'998.00 |     388'210.85 |     20'299.91 |
| generating large command graphs for N nodes - 1                                                                                                   | contracting tree topology        |     425'021.00 |     554'943.90 |    186'438.55 |
| generating large command graphs for N nodes - 1                                                                                                   | wave\_sim topology                |   2'001'910.00 |   2'190'300.28 |    111'627.81 |
| generating large command graphs for N nodes - 1                                                                                                   | jacobi topology                  |     740'798.00 |     814'585.65 |     26'328.09 |
| generating large command graphs for N nodes - 4                                                                                                   | soup topology                    |  37'693'662.00 |  42'708'788.49 |  1'317'030.91 |
| generating large command graphs for N nodes - 4                                                                                                   | chain topology                   |   2'892'143.00 |   3'289'545.35 |    120'696.77 |
| generating large command graphs for N nodes - 4                                                                                                   | expanding tree topology          |   6'472'217.00 |   6'558'322.48 |    354'070.33 |
| generating large command graphs for N nodes - 4                                                                                                   | contracting tree topology        |   3'705'610.00 |   3'727'709.31 |     10'797.28 |
| generating large command graphs for N nodes - 4                                                                                                   | wave\_sim topology                |  13'468'798.00 |  15'267'931.61 |    219'256.59 |
| generating large command graphs for N nodes - 4                                                                                                   | jacobi topology                  |   4'640'136.00 |   5'068'313.06 |    276'244.43 |
| generating large command graphs for N nodes - 16                                                                                                  | soup topology                    | 134'677'495.00 | 140'222'187.18 |  4'792'785.64 |
| generating large command graphs for N nodes - 16                                                                                                  | chain topology                   | 347'002'329.00 | 373'793'332.57 | 12'892'767.91 |
| generating large command graphs for N nodes - 16                                                                                                  | expanding tree topology          | 369'379'812.00 | 401'174'022.67 |  9'709'571.33 |
| generating large command graphs for N nodes - 16                                                                                                  | contracting tree topology        | 118'082'768.00 | 125'188'990.75 |  2'914'371.90 |
| generating large command graphs for N nodes - 16                                                                                                  | wave\_sim topology                | 117'436'908.00 | 124'911'858.59 |  4'357'252.17 |
| generating large command graphs for N nodes - 16                                                                                                  | jacobi topology                  | 111'881'261.00 | 120'284'391.53 |  2'434'633.13 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | soup topology                    |  16'075'192.00 |  17'141'495.17 |    974'101.84 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | chain topology                   |     280'999.00 |     283'813.45 |      2'970.96 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | expanding tree topology          |     345'160.00 |     380'102.28 |     25'475.26 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | contracting tree topology        |     491'637.00 |     495'239.03 |      4'144.94 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | wave\_sim topology                |   2'002'862.00 |   2'238'254.68 |    115'236.87 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | jacobi topology                  |     829'566.00 |     835'094.97 |      7'675.01 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | soup topology                    |  27'478'189.00 |  36'913'587.08 |  4'635'545.41 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | chain topology                   |     570'016.00 |     816'777.54 |    145'587.63 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | expanding tree topology          |     700'622.00 |   1'114'400.07 |    207'545.77 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | contracting tree topology        |     692'156.00 |     993'383.09 |    113'582.30 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | wave\_sim topology                |   4'067'194.00 |   6'086'953.73 |  1'723'257.20 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | jacobi topology                  |   1'645'157.00 |   2'274'749.24 |    433'541.99 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | soup topology                    |  18'305'446.00 |  21'162'705.45 |    455'990.50 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | chain topology                   |     579'553.00 |     589'640.81 |     27'401.21 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | expanding tree topology          |     707'946.00 |     711'322.36 |      5'283.64 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | contracting tree topology        |     738'072.00 |     779'665.33 |     31'784.81 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | wave\_sim topology                |   4'080'819.00 |   4'317'302.69 |     55'169.54 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | jacobi topology                  |   1'259'248.00 |   1'353'882.87 |     50'593.27 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | soup topology                    |  17'750'267.00 |  36'455'856.56 |  7'625'407.37 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | chain topology                   |   1'021'459.00 |   1'640'216.59 |    425'177.35 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | expanding tree topology          |   1'300'535.00 |   1'965'087.40 |    458'521.03 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | contracting tree topology        |   1'303'009.00 |   2'038'953.37 |    365'667.13 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | wave\_sim topology                |   6'875'429.00 |  11'350'992.55 |  2'318'000.42 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | jacobi topology                  |   2'208'921.00 |   3'386'860.19 |    591'689.43 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | soup topology                    |  38'191'833.00 |  43'536'383.12 |  2'002'852.88 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | chain topology                   |   2'971'893.00 |   3'452'317.08 |    360'511.71 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | expanding tree topology          |   5'772'705.00 |   6'639'552.70 |    517'593.65 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | contracting tree topology        |   3'286'228.00 |   3'758'447.01 |    147'103.69 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | wave\_sim topology                |  13'711'105.00 |  15'552'143.29 |    698'281.69 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | jacobi topology                  |   4'732'069.00 |   5'407'416.58 |    427'318.88 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | soup topology                    |  60'487'451.00 |  76'309'344.95 |  5'795'827.28 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | chain topology                   |   4'889'347.00 |   5'695'196.21 |    710'153.99 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | expanding tree topology          |   8'022'707.00 |   9'105'694.96 |    647'633.55 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | contracting tree topology        |   5'163'175.00 |   6'353'015.36 |    831'660.33 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | wave\_sim topology                |  24'033'350.00 |  29'848'573.93 |  2'919'235.36 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | jacobi topology                  |   5'125'583.00 |   6'650'958.39 |  1'336'018.68 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | soup topology                    |  40'914'607.00 |  45'767'725.38 |    933'872.81 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | chain topology                   |   3'291'047.00 |   3'693'096.09 |    126'866.15 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | expanding tree topology          |   6'942'436.00 |   6'967'677.07 |     11'736.30 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | contracting tree topology        |   3'593'659.00 |   4'072'027.23 |    138'554.89 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | wave\_sim topology                |  15'790'995.00 |  17'586'079.53 |    333'177.95 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | jacobi topology                  |   5'222'636.00 |   5'846'439.62 |    150'370.80 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | soup topology                    |  35'933'315.00 |  53'697'878.12 | 10'651'894.65 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | chain topology                   |   4'588'889.00 |   6'255'082.67 |    860'749.88 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | expanding tree topology          |   7'248'734.00 |   8'744'404.08 |    969'594.78 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | contracting tree topology        |   4'279'875.00 |   6'118'099.03 |  1'058'305.10 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | wave\_sim topology                |  23'307'919.00 |  29'588'274.68 |  3'760'384.05 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | jacobi topology                  |   7'851'723.00 |  11'271'631.21 |  1'498'389.07 |

All numbers are in nanoseconds.