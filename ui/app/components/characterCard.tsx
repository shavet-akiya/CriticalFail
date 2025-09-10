function CharacterCard() {
  return (
    <div className="card bg-base-100 w-96 shadow-sm">
      <div>
        <img
          src="https://img.daisyui.com/images/stock/photo-1606107557195-0e29a4b5b4aa.webp"
          alt="Shoes"
        />
      </div>
      <div className="card-body">
        <h2 className="card-title">Character Name</h2>
        <p>Class/Race</p>
        <p>HP</p>
        <p></p>
        <div className="card-actions justify-end">
          <button className="btn btn-primary">Buy Now</button>
        </div>
      </div>
    </div>
  );
}

export default CharacterCard;
