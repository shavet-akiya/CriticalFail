// react suspense - promise 

export default function Loading() {
    return (
        <div className="flex flex-col justify-center items-center h-screen">
            <span className="loading loading-ring loading-xl obsidian-colour"></span>
            <h1 className="obsidian-colour text-2xl pt-8">Waiting</h1>
        </div>
    )
}