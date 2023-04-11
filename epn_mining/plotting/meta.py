freqs_filename = {0: u'0-200MHz', # in MHz
                  1: u'200-400MHz',
                  2: u'400-700MHz',
                  3: u'700-1000MHz',
                  4: u'1000-1500MHz',
                  5: u'1500-2000MHz',
                  6: u'2000+MHz'}

palette_colourlovers = {
    0: '#490A3D',
    1: '#BD1550',
    2: '#E97F02',
    3: '#F8CA00',
    4: '#8A9B0F',

} # https://www.colourlovers.com/palette/848743/(_”_)

palette_qualitative_paired_12 = {i:c for i, c in enumerate(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                                      '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                                      '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']) } # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12

palette_Geys = {i:c for i, c in enumerate(
    ['#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#252525', '#737373',
     '#000000'][::-1]
)} # https://colorbrewer2.org/#type=sequential&scheme=Greys&n=8

palette_rankin_class = {i:c for i, c in enumerate(
        [
#             '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'
#             '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'
            '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99'
        ]
    )
}

palettes = {
    'colourlovers': palette_colourlovers,
    'qualitative_paired_12': palette_qualitative_paired_12,
    'Greys': palette_Geys,
    'rankin': palette_rankin_class,
}
# palette = {i:c for i, c in enumerate(['#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
#                                       '#80b1d3', '#fdb462', '#b3de69', '#fccde5']) } # https://colorbrewer2.org/#type=qualitative&scheme=Set3&n=8

# palette = {i:c for i, c in enumerate(
#     ['#ffffff', '#fdbf6f', '#f0f0f0', '#bdbdbd', '#d9d9d9',
#      '#737373', '#969696', '#525252', '#252525'][::-1]
# )} # https://colorbrewer2.org/#type=sequential&scheme=Greys&n=8


