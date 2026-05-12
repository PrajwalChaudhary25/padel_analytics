import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # saves file without needing a display window

df = pd.read_csv("output/shots.csv")

# keep only main two players
df = df[df["player_id"].isin([0, 1])]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Padel Match — Shot Analytics", fontsize=14, fontweight="bold")

# chart 1 — overall shot type breakdown
shot_counts = df["shot_type"].value_counts()
colors = ["#2ecc71", "#e67e22", "#e74c3c"]
axes[0].bar(shot_counts.index, shot_counts.values, color=colors)
axes[0].set_title("Total shots by type")
axes[0].set_ylabel("Count")
for i, v in enumerate(shot_counts.values):
    axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

# chart 2 — per player shot breakdown
player_shots = df.groupby(["player_id", "shot_type"]).size().unstack(fill_value=0)
player_shots.plot(kind="bar", ax=axes[1], color=colors, rot=0)
axes[1].set_title("Shots per player")
axes[1].set_xlabel("Player ID")
axes[1].set_ylabel("Count")
axes[1].legend(title="Shot type")

# chart 3 — shot timeline (when shots happen across the match)
for pid, color in zip([0, 1], ["#3498db", "#e74c3c"]):
    player_df = df[df["player_id"] == pid]
    axes[2].scatter(player_df["timestamp"], 
                    [pid] * len(player_df),
                    c=color, alpha=0.4, s=15,
                    label=f"Player {pid}")
axes[2].set_title("Shot timeline")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(["Player 0", "Player 1"])
axes[2].legend()

plt.tight_layout()
plt.savefig("output/analytics.png", dpi=150, bbox_inches="tight")
print("Chart saved to output/analytics.png")