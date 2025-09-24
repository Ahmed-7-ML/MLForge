# Matplotlib
import matplotlib.pyplot as plt

def scatter(x_data, y_data, x_label='x_data', y_label='Y_data'):
    plt.scatter(x_data, y_data, c='red', alpha=0.5 , s= 120)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{x_data} VS {y_data}')
    return plt.show()

def line(x_data, y_data, x_label='x_data', y_label='Y_data') :
    plt.plot(x_data, y_data, lw = 3)
    plt.title(f'{x_label} VS {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels = x_data, loc ='upper right')
    return plt.show()

def bar(x_data, y_data, x_label='x_data', y_label='Y_data') :
    plt.bar(x_data, y_data, color='red', aligh ='center', width=0.5, edgecolor ='blue', lw=3)
    plt.title(f'{x_label} VS {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.show()

def pie(x_data, y_data, x_label='x_data', y_label='Y_data') :
    plt.pie(x_data, labels=y_data, color='red', autopct='%.2f%%', startangle=90, lw=3)
    plt.tight_layout()
    plt.title(f'{x_label} VS {y_label}')
    return plt.show()

def hist(ages, title ='ages'):
    plt.hist(ages, bins=10, cumulative=True)
    plt.title(title)
    plt.show()

def boxplot(ages, title = 'ages'):
    plt.boxplot(ages, bins=10, cumulative=True)
    plt.title(title)
    plt.show()

# seaborn
import seaborn as sns

#Pair plot
def pairplot(df, x) :
    return sns.pairplot(df, hue=x)

#histogram
def hist(df, ages, title ='ages'):
    sns.histplot(df,ages, bins=10,multible = 'stack' , stat= 'percent', alpha= 0.2, kde=True, palette='pastel')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(title)
    return plt.show()

#boxplot
def boxplot(df, ages, title = 'ages' ):
    sns.boxenplot(x=ages, data=df, y='time')
    plt.title(title)
    return plt.show()

#Countplot
def countplot(df , x_axis, hu, huorder, ord):
    sns.countplot(x=x_axis, data=df, hue=hu, hue_order=huorder , order=ord)
    plt.title('count plot')
    return plt.show()

def heatmap(df):
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('carrelation heatmap')
    return plt.show()



