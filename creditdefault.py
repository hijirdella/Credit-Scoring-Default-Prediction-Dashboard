# Combined visualization: Default Rate and Cumulative Default Rate by Loan Amount Decile

# Set style
sns.set_style("whitegrid")

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8,5))

# Bar plot for default rate (primary y-axis)
bars = ax1.bar(
    decile_df['decile'],
    decile_df['default_rate'],
    color="#FF6D00",      # dark orange
    edgecolor='black',
    alpha=0.8,
    label='Default Rate (%)'
)

ax1.set_xlabel('Loan Amount Decile (1 = smallest loans)')
ax1.set_ylabel('Default Rate (%)', color="#FF6D00")
ax1.tick_params(axis='y', labelcolor="#FF6D00")

# Create secondary axis for cumulative default rate
ax2 = ax1.twinx()

# Line plot for cumulative default rate (secondary y-axis)
ax2.plot(
    decile_df['decile'],
    decile_df['cumulative_default_rate'],
    marker='o',
    linestyle='-',
    color="#FFD180",      # light orange
    label='Cumulative Default Rate (%)'
)

ax2.set_ylabel('Cumulative Default Rate (%)', color="#FFD180")
ax2.tick_params(axis='y', labelcolor="#FFD180")

# Title and layout
plt.title('Default and Cumulative Default Rate by Loan Amount Decile')
fig.tight_layout()

# Build combined legend
lines_labels = []
for ax in [ax1, ax2]:
    line, label = ax.get_legend_handles_labels()
    lines_labels += list(zip(line, label))

# Remove duplicates and show legend
handles, labels = zip(*dict(lines_labels).items())
ax1.legend(handles, labels, loc='upper left')

plt.show()
