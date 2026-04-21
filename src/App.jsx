import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);

  const search = async () => {
    const res = await fetch(`http://localhost:8000/query?q=${query}`);
    const json = await res.json();
    setResult(json);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>🍽 AI Nutrition Dashboard</h2>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Try: top protein foods"
      />

      <button onClick={search}>Search</button>

      <div style={{ marginTop: 20 }}>

        {/* TEXT */}
        {result?.type === "text" &&
          result.data.map((item, i) => (
            <div key={i}>
              <h4>{item.food_name}</h4>
              <p>Calories: {item.calories}</p>
              <p>Protein: {item.protein}</p>
            </div>
          ))}

        {/* CHART */}
        {result?.type === "chart" && (
          <div>
            <img
              src={`http://localhost:8000/${result.chart_path}`}
              width="500"
              alt="chart"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;