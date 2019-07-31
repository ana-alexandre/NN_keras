#plots single or stacked histograms
#note: stack hist is made in plan for the resolution histograms

def hist(array, xlabel, ylabel = '', title = '', directory = '',color = 'b'):
    #this function plots individual histograms of each of the variables

    #file to save histogram to
    file_name = directory + 'hist_' + xlabel + '.png'

    #plot the histogram and save to file
    plt.hist(array, bins=100,color = color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def stack_hist(arrays, xlabel, ylabel = '', title = '', directory = '', xmin=-10, xmax=2, legend = ['1','2','3'],loc='upper right'):
    #plot stacked histogram of multiple variables
    #file to save histogram to
    file_name = directory + 'hist_' + xlabel + '.png'

    fig = plt.figure()

    #plot the histogram and save to file
    plt.hist(arrays, 500, stacked=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin,xmax)
    plt.legend(legend, loc=loc)
    fig.savefig(title+'.png')
    plt.show()



