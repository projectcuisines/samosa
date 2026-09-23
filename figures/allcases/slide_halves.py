from matplotlib.transforms import Bbox

# Slide copies of the stacked figures that set all cases above common cases.
# Each block is saved to its own PNG, cropped from the figure as drawn so that
# it matches the paper exactly. Not used by the manuscript.
#
# A figure built from subfigures is split along them. Otherwise every axes and
# figure text goes to the block on its side of the widest gap between the rows
# of axes, which always falls between the two blocks.

def save_halves( fig, outname, tags=( 'all', 'common' ), dpi=200, pad=0.1 ):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if fig.subfigs:
        for tag, sub in zip( tags, fig.subfigs ):
            bb = sub.get_tightbbox( renderer ).transformed( fig.dpi_scale_trans.inverted() ).padded( pad )
            fig.savefig( f'{outname}_{tag}.png', bbox_inches=bb, dpi=dpi )
        return

    spans = sorted( ( ax.get_position().y0, ax.get_position().y1 ) for ax in fig.axes )
    reach, widest, mid = spans[ 0 ][ 1 ], 0.0, None
    for y0, y1 in spans[ 1: ]:
        if y0 - reach > widest:
            widest, mid = y0 - reach, ( y0 + reach )/2
        reach = max( reach, y1 )
    mid *= fig.bbox.height

    upper, lower = [], []
    for artist in list( fig.axes ) + list( fig.texts ):
        bb = artist.get_tightbbox( renderer )
        if artist.get_visible() and bb is not None and bb.width > 0:
            ( upper if ( bb.y0 + bb.y1 )/2 > mid else lower ).append( bb )

    for tag, boxes in zip( tags, ( upper, lower ) ):
        bb = Bbox.union( boxes ).transformed( fig.dpi_scale_trans.inverted() ).padded( pad )
        fig.savefig( f'{outname}_{tag}.png', bbox_inches=bb, dpi=dpi )
