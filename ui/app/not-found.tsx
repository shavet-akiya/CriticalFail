import Link from 'next/link'

export default function NotFound() {
    return (
        <div className='select-none flex flex-col items-center justify-center'>
            <p className='obsidian-colour text-5xl pb-4'>You've hit a dead end!</p>
            <p className='text-xl text-gray-400 pb-16'>You must find your way home.</p>
            <button>
                <Link href="/" className='btn btn-primary'>Return Home</Link>
            </button>
        </div>
    )
}