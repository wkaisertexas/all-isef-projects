/* CALLBACK: category_filter */
const data = source.data;
const category = data["category"];

if (this.value.length === 0) {
  for (let i = 0; i < category.length; i++) {
    data["x"][i] = data["x_back"][i];
    data["y"][i] = data["y_back"][i];
  }
  source.change.emit();
  return;
}

for (let i = 0; i < category.length; i++) {
  let include = false;
  for (let j = 0; j < this.value.length; j++) {
    if (this.value[j] === category[i]) {
      include = true;
      break;
    }
  }
  if (include) {
    data["x"][i] = data["x_back"][i];
    data["y"][i] = data["y_back"][i];
  } else {
    data["x"][i] = undefined;
    data["y"][i] = undefined;
  }
}
source.change.emit();
/* END_CALLBACK */

/* CALLBACK: winner_filter */
const data = source.data;
const numAwards = data["num_awards"];

for (let i = 0; i < numAwards.length; i++) {
  if (this.active && numAwards[i] <= 0) {
    data["x"][i] = undefined;
    data["y"][i] = undefined;
  } else {
    data["x"][i] = data["x_back"][i];
    data["y"][i] = data["y_back"][i];
  }
}

source.change.emit();
/* END_CALLBACK */

/* CALLBACK: search_filter */
const data = source.data;
const title = data["title"];
const abstract = data["abstract"];
const parts = this.value.toLowerCase().split(" ").filter(Boolean);

if (parts.length === 0) {
  for (let i = 0; i < title.length; i++) {
    data["x"][i] = data["x_back"][i];
    data["y"][i] = data["y_back"][i];
  }
  source.change.emit();
  return;
}

for (let i = 0; i < title.length; i++) {
  const projectTitle = String(title[i] ?? "").toLowerCase();
  const projectAbstract = String(abstract[i] ?? "").toLowerCase();

  let include = false;
  for (let j = 0; j < parts.length; j++) {
    const part = parts[j];
    if (projectTitle.includes(part) || projectAbstract.includes(part)) {
      include = true;
      break;
    }
  }

  if (include) {
    data["x"][i] = data["x_back"][i];
    data["y"][i] = data["y_back"][i];
  } else {
    data["x"][i] = undefined;
    data["y"][i] = undefined;
  }
}
source.change.emit();
/* END_CALLBACK */
