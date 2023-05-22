pink = (230/255, 29/255, 95/255, 1)
pink_translucid = (230/255, 29/255, 95/255, .2)
blue = (47/255, 161/255, 214/255, 0.2)
blue_full = (47/255, 161/255, 214/255, 1)

freqs_filename = {0: u'0-200MHz', # in MHz
                  1: u'200-400MHz',
                  2: u'400-700MHz',
                  3: u'700-1000MHz',
                  4: u'1000-1500MHz',
                  5: u'1500-2000MHz',
                  6: u'2000+MHz'}

palette = {
    0: '#490A3D',
    1: '#BD1550',
    2: '#E97F02',
    3: '#F8CA00',
    4: '#8A9B0F',

} # https://www.colourlovers.com/palette/848743/(_”_)

palette = {i:c for i, c in enumerate(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                                      '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                                      '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']) } # https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12

palette_rankin_class = {i:c for i, c in enumerate(
        [ '#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc' ]
    )
}

palette_morph_class = {
    # c:palette_rankin_class[i] for i, c in enumerate(
    #     np.unique(morphological_classes[(morphological_classes != 'N/A')])
    # )
    'Core single': palette_rankin_class[0],
    'Conal single': palette_rankin_class[1],
    'Conal double': palette_rankin_class[2],
    'Triple': palette_rankin_class[3],
    'Multiple': palette_rankin_class[4],
}
palette_morph_class['N/A'] = '#000000'

palette = {i:c for i, c in enumerate(
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

